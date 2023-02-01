"""
All code is from https://github.com/speechbrain/speechbrain

SepformerSeparation: speechbrain/pretrained/interface/SepformerSeparation
encoder            : speechbrain/lobe/models/dual_path/Encoder
masknet            : speechbrain/lobe/models/dual_path/SepformerWrapper
decoder            : speechbrain/lobe/models/dual_path/Decoder
param              : speechbrain/recipes/WSJ0Mix/separation/hparams/sepformer.yaml

Apache License 2.0
"""

import copy
import torch
import torch.nn
import torch.nn.functional as F
from .Transformer import TransformerEncoder, PositionalEncoding
from .basic import RNN, Linear

class SepformerSeparation(torch.nn.Module):
    """The wrapper for the sepformer model which combines the Encoder, Masknet and the decoder
    https://arxiv.org/abs/2010.13154

    Arguments
    ---------

    encoder_kernel_size: int,
        The kernel size used in the encoder
    encoder_in_nchannels: int,
        The number of channels of the input audio
    encoder_out_nchannels: int,
        The number of filters used in the encoder.
        Also, number of channels that would be inputted to the intra and inter blocks.
    masknet_chunksize: int,
        The chunk length that is to be processed by the intra blocks
    masknet_numlayers: int,
        The number of layers of combination of inter and intra blocks
    masknet_norm: str,
        The normalization type to be used in the masknet
        Should be one of 'ln' -- layernorm, 'gln' -- globallayernorm
                         'cln' -- cumulative layernorm, 'bn' -- batchnorm
                         -- see the select_norm function above for more details
    masknet_useextralinearlayer: bool,
        Whether or not to use a linear layer at the output of intra and inter blocks
    masknet_extraskipconnection: bool,
        This introduces extra skip connections around the intra block
    masknet_numspks: int,
        This determines the number of speakers to estimate
    intra_numlayers: int,
        This determines the number of layers in the intra block
    inter_numlayers: int,
        This determines the number of layers in the inter block
    intra_nhead: int,
        This determines the number of parallel attention heads in the intra block
    inter_nhead: int,
        This determines the number of parallel attention heads in the inter block
    intra_dffn: int,
        The number of dimensions in the positional feedforward model in the inter block
    inter_dffn: int,
        The number of dimensions in the positional feedforward model in the intra block
    intra_use_positional: bool,
        Whether or not to use positional encodings in the intra block
    inter_use_positional: bool,
        Whether or not to use positional encodings in the inter block
    intra_norm_before: bool
        Whether or not we use normalization before the transformations in the intra block
    inter_norm_before: bool
        Whether or not we use normalization before the transformations in the inter block

    Example
    -----
    >>> model = SepformerWrapper()
    >>> inp = torch.rand(1, 160)
    >>> result = model.forward(inp)
    >>> result.shape
    torch.Size([1, 160, 2])
    """

    def __init__(
        self,
        encoder_kernel_size=16,
        encoder_in_nchannels=1,
        encoder_out_nchannels=256,
        masknet_chunksize=250,
        masknet_numlayers=2,
        masknet_norm="ln",
        masknet_useextralinearlayer=False,
        masknet_extraskipconnection=True,
        masknet_numspks=2,
        intra_numlayers=8,
        inter_numlayers=8,
        intra_nhead=8,
        inter_nhead=8,
        intra_dffn=1024,
        inter_dffn=1024,
        intra_use_positional=True,
        inter_use_positional=True,
        intra_norm_before=True,
        inter_norm_before=True,
    ):

        super(SepformerSeparation, self).__init__()
        self.encoder = Encoder(
            kernel_size=encoder_kernel_size,
            out_channels=encoder_out_nchannels,
            in_channels=encoder_in_nchannels,
        )
        intra_model = SBTransformerBlock(
            num_layers=intra_numlayers,
            d_model=encoder_out_nchannels,
            nhead=intra_nhead,
            d_ffn=intra_dffn,
            use_positional_encoding=intra_use_positional,
            norm_before=intra_norm_before,
        )

        inter_model = SBTransformerBlock(
            num_layers=inter_numlayers,
            d_model=encoder_out_nchannels,
            nhead=inter_nhead,
            d_ffn=inter_dffn,
            use_positional_encoding=inter_use_positional,
            norm_before=inter_norm_before,
        )

        self.masknet = Dual_Path_Model(
            in_channels=encoder_out_nchannels,
            out_channels=encoder_out_nchannels,
            intra_model=intra_model,
            inter_model=inter_model,
            num_layers=masknet_numlayers,
            norm=masknet_norm,
            K=masknet_chunksize,
            num_spks=masknet_numspks,
            skip_around_intra=masknet_extraskipconnection,
            linear_layer_after_inter_intra=masknet_useextralinearlayer,
        )
        self.decoder = Decoder(
            in_channels=encoder_out_nchannels,
            out_channels=encoder_in_nchannels,
            kernel_size=encoder_kernel_size,
            stride=encoder_kernel_size // 2,
            bias=False,
        )
        self.num_spks = masknet_numspks

        # reinitialize the parameters
        for module in [self.encoder, self.masknet, self.decoder]:
            self.reset_layer_recursively(module)

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the network"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def forward(self, mix):
        """ Processes the input tensor x and returns an output tensor."""
        mix_w = self.encoder(mix)
        est_mask = self.masknet(mix_w)
        mix_w = torch.stack([mix_w] * self.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        print(T_origin, T_est)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source


class Encoder(torch.nn.Module):
    """Convolutional Encoder Layer.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.

    Example
    -------
    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape
    torch.Size([2, 64, 499])
    """

    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super(Encoder, self).__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x):
        """Return the encoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, L].
        Return
        ------
        x : torch.Tensor
            Encoded tensor with dimensionality [B, N, T_out].

        where B = Batchsize
              L = Number of timepoints
              N = Number of filters
              T_out = Number of timepoints at the output of the encoder
        """
        # B x L -> B x 1 x L
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        # B x 1 x L -> B x N x T_out
        x = self.conv1d(x)
        x = F.relu(x)

        return x


class Decoder(torch.nn.ConvTranspose1d):
    """A decoder layer that consists of ConvTranspose1d.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.


    Example
    ---------
    >>> x = torch.randn(2, 100, 1000)
    >>> decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    >>> h = decoder(x)
    >>> h.shape
    torch.Size([2, 1003])
    """

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """Return the decoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, N, L].
                where, B = Batchsize,
                       N = number of filters
                       L = time points
        """

        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} accept 3/4D tensor as input".format(self.__name__)
            )
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class Dual_Path_Model(torch.nn.Module):
    """The dual path model which is the basis for dualpathrnn, sepformer, dptnet.

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    intra_model : torch.nn.module
        Model to process within the chunks.
    inter_model : torch.nn.module
        model to process across the chunks,
    num_layers : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    K : int
        Chunk length.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        Skip connection around intra.
    linear_layer_after_inter_intra : bool
        Linear layer after inter and intra.
    use_global_pos_enc : bool
        Global positional encodings.
    max_length : int
        Maximum sequence length.

    Example
    ---------
    >>> intra_block = SBTransformerBlock(1, 64, 8)
    >>> inter_block = SBTransformerBlock(1, 64, 8)
    >>> dual_path_model = Dual_Path_Model(64, 64, intra_block, inter_block, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = dual_path_model(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        inter_model,
        num_layers=1,
        norm="ln",
        K=200,
        num_spks=2,
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
        use_global_pos_enc=False,
        max_length=20000,
    ):
        super(Dual_Path_Model, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = PositionalEncoding(max_length)

        self.dual_mdl = torch.nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Dual_Computation_Block(
                        intra_model,
                        inter_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                        linear_layer_after_inter_intra=linear_layer_after_inter_intra,
                    )
                )
            )

        self.conv2d = torch.nn.Conv2d(
            out_channels, out_channels * num_spks, kernel_size=1
        )
        self.end_conv1x1 = torch.nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = torch.nn.PReLU()
        self.activation = torch.nn.ReLU()
        # gated output layer
        self.output = torch.nn.Sequential(
            torch.nn.Conv1d(out_channels, out_channels, 1), torch.nn.Tanh()
        )
        self.output_gate = torch.nn.Sequential(
            torch.nn.Conv1d(out_channels, out_channels, 1), torch.nn.Sigmoid()
        )

    def forward(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, L].

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [spks, B, N, L]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               L = the number of time points
        """

        # before each line we indicate the shape after executing the line

        # [B, N, L]
        x = self.norm(x)

        # [B, N, L]
        x = self.conv1d(x)
        if self.use_global_pos_enc:
            x = self.pos_enc(x.transpose(1, -1)).transpose(1, -1) + x * (
                x.size(1) ** 0.5
            )

        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)

        # [B, N, K, S]
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x)
        x = self.prelu(x)

        # [B, N*spks, K, S]
        x = self.conv2d(x)
        B, _, K, S = x.shape

        # [B*spks, N, K, S]
        x = x.view(B * self.num_spks, -1, K, S)

        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)

        # [B*spks, N, L]
        x = self.end_conv1x1(x)

        # [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x

    def _padding(self, input, K):
        """Padding the audio times.

        Arguments
        ---------
        K : int
            Chunks of length.
        P : int
            Hop size.
        input : torch.Tensor
            Tensor of size [B, N, L].
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = (
                torch.Tensor(torch.zeros(B, N, gap))
                .type(input.dtype)
                .to(input.device)
            )
            input = torch.cat([input, pad], dim=2)

        _pad = (
            torch.Tensor(torch.zeros(B, N, P))
            .type(input.dtype)
            .to(input.device)
        )
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """The segmentation stage splits

        Arguments
        ---------
        K : int
            Length of the chunks.
        input : torch.Tensor
            Tensor with dim [B, N, L].

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = (
            torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)
        )

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """Merge the sequence with the overlap-and-add method.

        Arguments
        ---------
        input : torch.tensor
            Tensor with dim [B, N, K, S].
        gap : int
            Padding length.

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, L].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points

        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class SBTransformerBlock(torch.nn.Module):
    """A wrapper for the SpeechBrain implementation of the transformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of layers.
    d_model : int
        Dimensionality of the representation.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Dimensionality of positional feed forward.
    input_shape : tuple
        Shape of input.
    kdim : int
        Dimension of the key (Optional).
    vdim : int
        Dimension of the value (Optional).
    dropout : float
        Dropout rate.
    activation : str
        Activation function.
    use_positional_encoding : bool
        If true we use a positional encoding.
    norm_before: bool
        Use normalization before transformations.

    Example
    ---------
    >>> x = torch.randn(10, 100, 64)
    >>> block = SBTransformerBlock(1, 64, 8)
    >>> x = block(x)
    >>> x.shape
    torch.Size([10, 100, 64])
    """

    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        d_ffn=2048,
        input_shape=None,
        kdim=None,
        vdim=None,
        dropout=0.1,
        activation="relu",
        use_positional_encoding=False,
        norm_before=False,
        attention_type="regularMHA",
    ):
        super(SBTransformerBlock, self).__init__()
        self.use_positional_encoding = use_positional_encoding

        if activation == "relu":
            activation = torch.nn.ReLU
        elif activation == "gelu":
            activation = torch.nn.GELU
        else:
            raise ValueError("unknown activation")

        self.mdl = TransformerEncoder(
            num_layers=num_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            input_shape=input_shape,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
            normalize_before=norm_before,
            attention_type=attention_type,
        )

        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(input_size=d_model)

    def forward(self, x):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters

        """
        if self.use_positional_encoding:
            pos_enc = self.pos_enc(x)
            return self.mdl(x + pos_enc)[0]
        else:
            return self.mdl(x)[0]


class Dual_Computation_Block(torch.nn.Module):
    """Computation block for dual-path processing.

    Arguments
    ---------
    intra_mdl : torch.nn.module
        Model to process within the chunks.
     inter_mdl : torch.nn.module
        Model to process across the chunks.
     out_channels : int
        Dimensionality of inter/intra model.
     norm : str
        Normalization type.
     skip_around_intra : bool
        Skip connection around the intra layer.
     linear_layer_after_inter_intra : bool
        Linear layer or not after inter or intra.

    Example
    ---------
        >>> intra_block = SBTransformerBlock(1, 64, 8)
        >>> inter_block = SBTransformerBlock(1, 64, 8)
        >>> dual_comp_block = Dual_Computation_Block(intra_block, inter_block, 64)
        >>> x = torch.randn(10, 64, 100, 10)
        >>> x = dual_comp_block(x)
        >>> x.shape
        torch.Size([10, 64, 100, 10])
    """

    def __init__(
        self,
        intra_mdl,
        inter_mdl,
        out_channels,
        norm="ln",
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
    ):
        super(Dual_Computation_Block, self).__init__()

        self.intra_mdl = intra_mdl
        self.inter_mdl = inter_mdl
        self.skip_around_intra = skip_around_intra
        self.linear_layer_after_inter_intra = linear_layer_after_inter_intra

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 4)
            self.inter_norm = select_norm(norm, out_channels, 4)

        # Linear
        if linear_layer_after_inter_intra:
            if isinstance(intra_mdl, SBRNNBlock):
                self.intra_linear = Linear(
                    out_channels, input_size=2 * intra_mdl.mdl.rnn.hidden_size
                )
            else:
                self.intra_linear = Linear(
                    out_channels, input_size=out_channels
                )

            if isinstance(inter_mdl, SBRNNBlock):
                self.inter_linear = Linear(
                    out_channels, input_size=2 * intra_mdl.mdl.rnn.hidden_size
                )
            else:
                self.inter_linear = Linear(
                    out_channels, input_size=out_channels
                )

    def forward(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, K, S].


        Return
        ---------
        out: torch.Tensor
            Output tensor of dimension [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
        """
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        # [BS, K, H]

        intra = self.intra_mdl(intra)

        # [BS, K, N]
        if self.linear_layer_after_inter_intra:
            intra = self.intra_linear(intra)

        # [B, S, K, N]
        intra = intra.view(B, S, K, N)
        # [B, N, K, S]
        intra = intra.permute(0, 3, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)

        # [B, N, K, S]
        if self.skip_around_intra:
            intra = intra + x

        # inter RNN
        # [BK, S, N]
        inter = intra.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        # [BK, S, H]
        inter = self.inter_mdl(inter)

        # [BK, S, N]
        if self.linear_layer_after_inter_intra:
            inter = self.inter_linear(inter)

        # [B, K, S, N]
        inter = inter.view(B, K, S, N)
        # [B, N, K, S]
        inter = inter.permute(0, 3, 1, 2).contiguous()
        if self.norm is not None:
            inter = self.inter_norm(inter)
        # [B, N, K, S]
        out = inter + intra

        return out


class SBRNNBlock(torch.nn.Module):
    """RNNBlock for the dual path pipeline.

    Arguments
    ---------
    input_size : int
        Dimensionality of the input features.
    hidden_channels : int
        Dimensionality of the latent layer of the rnn.
    num_layers : int
        Number of the rnn layers.
    rnn_type : str
        Type of the the rnn cell.
    dropout : float
        Dropout rate
    bidirectional : bool
        If True, bidirectional.

    Example
    ---------
    >>> x = torch.randn(10, 100, 64)
    >>> rnn = SBRNNBlock(64, 100, 1, bidirectional=True)
    >>> x = rnn(x)
    >>> x.shape
    torch.Size([10, 100, 200])
    """

    def __init__(
        self,
        input_size,
        hidden_channels,
        num_layers,
        rnn_type="LSTM",
        dropout=0,
        bidirectional=True,
    ):
        super(SBRNNBlock, self).__init__()

        self.mdl = getattr(RNN, rnn_type)(
            hidden_channels,
            input_size=input_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            [B, L, N]
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """

        return self.mdl(x)[0]


def select_norm(norm, dim, shape, eps=1e-8):
    """Just a wrapper to select the normalization type.
    """

    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True, eps=eps)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True, eps=eps)
    if norm == "ln":
        return torch.nn.GroupNorm(1, dim, eps=eps)
    else:
        return torch.nn.BatchNorm1d(dim)


class GlobalLayerNorm(torch.nn.Module):
    """Calculate Global Layer Normalization.

    Arguments
    ---------
       dim : (int or list or torch.Size)
           Input shape from an expected input of size.
       eps : float
           A value added to the denominator for numerical stability.
       elementwise_affine : bool
          A boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> GLN = GlobalLayerNorm(10, 3)
    >>> x_norm = GLN(x)
    """

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = torch.nn.Parameter(torch.ones(self.dim, 1))
                self.bias = torch.nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = torch.nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = torch.nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of size [N, C, K, S] or [N, C, L].
        """
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)

        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(torch.nn.LayerNorm):
    """Calculate Cumulative Layer Normalization.

       Arguments
       ---------
       dim : int
        Dimension that you want to normalize.
       elementwise_affine : True
        Learnable per-element affine parameters.

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> CLN = CumulativeLayerNorm(10)
    >>> x_norm = CLN(x)
    """

    def __init__(self, dim, elementwise_affine=True, eps=1e-8):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=eps
        )

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor size [N, C, K, S] or [N, C, L]
        """
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            x = super().forward(x)
            # N x C x K x S
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x
"""
All code is from https://github.com/speechbrain/speechbrain

speechbrain/lobes/models/transfomer/Transformer
speechbrain/nnet/attention

Library implementing normalization.

Authors
 * Mirco Ravanelli 2020
 * Guillermo Cámbara 2021
 * Sarthak Yadav 2022
"""

import math
import numpy as np
from typing import Optional

import torch
import torch.nn
import torch.nn.functional as F
from .normalization import LayerNorm

class TransformerEncoder(torch.nn.Module):
    """This class implements the transformer encoder.

    Arguments
    ---------
    num_layers : int
        Number of transformer layers to include.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    d_model : int
        The dimension of the input embedding.
    kdim : int
        Dimension for key (Optional).
    vdim : int
        Dimension for value (Optional).
    dropout : float
        Dropout for the encoder (Optional).
    input_module: torch class
        The module to process the source input feature to expected
        feature dimension (Optional).

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoder(1, 8, 512, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        input_shape=None,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=torch.nn.ReLU,
        normalize_before=False,
        causal=False,
        layerdrop_prob=0.0,
        attention_type="regularMHA",
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                    attention_type=attention_type,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model, eps=1e-6)
        self.layerdrop_prob = layerdrop_prob
        self.rng = np.random.default_rng()

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : tensor
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
        output = src
        if self.layerdrop_prob > 0.0:
            keep_probs = self.rng.random(len(self.layers))
        else:
            keep_probs = None
        attention_lst = []
        for i, enc_layer in enumerate(self.layers):
            if (
                not self.training
                or self.layerdrop_prob == 0.0
                or keep_probs[i] > self.layerdrop_prob
            ):
                output, attention = enc_layer(
                    output,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos_embs=pos_embs,
                )

                attention_lst.append(attention)
        output = self.norm(output)
        return output, attention_lst

    
class TransformerEncoderLayer(torch.nn.Module):
    """This is an implementation of self-attention encoder layer.

    Arguments
    ----------
    d_ffn: int, optional
        The dimension of the feedforward network model hidden layer.
    nhead: int
        The number of heads in the multi-head attention models (default=8).
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    kdim: int, optional
        Dimension of the key.
    vdim: int, optional
        Dimension of the value.
    dropout: int, optional
        The dropout value.
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Netowrk layer,
        e.g., relu or gelu or swish.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoderLayer(512, 8, d_model=512)
    >>> output = net(x)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        nhead,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=torch.nn.ReLU,
        normalize_before=False,
        attention_type="regularMHA",
        causal=False,
    ):
        super().__init__()

        if attention_type == "regularMHA":
            self.self_att = MultiheadAttention(
                nhead=nhead,
                d_model=d_model,
                dropout=dropout,
                kdim=kdim,
                vdim=vdim,
            )

        elif attention_type == "RelPosMHAXL":
            self.self_att = RelPosMHAXL(
                d_model, nhead, dropout, mask_pos_future=causal
            )

        self.pos_ffn = PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

        self.norm1 = LayerNorm(d_model, eps=1e-6)
        self.norm2 = LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor
            The mask for the src query for each example in the batch.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys for each example in the batch.
        """

        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        output, self_attn = self.self_att(
            src1,
            src1,
            src1,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )

        # add & norm
        src = src + self.dropout1(output)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src1 = self.norm2(src)
        else:
            src1 = src
        output = self.pos_ffn(src1)

        # add & norm
        output = src + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)
        return output, self_attn


class PositionalEncoding(torch.nn.Module):
    """This class implements the absolute sinusoidal positional encoding function.

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))

    Arguments
    ---------
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).

    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = PositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float()
            * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments
        ---------
        x : tensor
            Input feature shape (batch, time, fea)
        """
        return self.pe[:, : x.size(1)].clone().detach()



class MultiheadAttention(torch.nn.Module):
    """ The class is a wrapper of MultiHead Attention for torch.nn.MultiHeadAttention.

    Reference: https://pytorch.org/docs/stable/nn.html

    Arguments
    ----------
    num_heads : int
        parallel attention heads.
    dropout : float
        a Dropout layer on attn_output_weights (default: 0.0).
    bias : bool
        add bias as module parameter (default: True).
    add_bias_kv : bool
        add bias to the key and value sequences at dim=0.
    add_zero_attn : bool
        add a new batch of zeros to the key and value sequences at dim=1.
    kdim : int
        total number of features in key (default: None).
    vdim : int
        total number of features in value (default: None).

    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = MultiheadAttention(nhead=8, d_model=inputs.shape[-1])
    >>> outputs, attn = net(inputs, inputs, inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        nhead,
        d_model,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
    ):
        super().__init__()

        self.att = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
        )

    def forward(
        self,
        query,
        key,
        value,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: Optional[torch.Tensor] = True,
        pos_embs: Optional[torch.Tensor] = None,
    ):
        """
        Arguments
        ----------
        query : torch.Tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        value : torch.Tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        key_padding_mask : torch.Tensor, optional
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        attn_mask : torch.Tensor, optional
            2D mask (L, S) where L is the target sequence length, S is
            the source sequence length.
            3D mask (N*num_heads, L, S) where N is the batch
            size, L is the target sequence length, S is the source sequence
            length. attn_mask ensure that position i is allowed to attend the
            unmasked positions. If a ByteTensor is provided, the non-zero
            positions are not allowed to attend while the zero positions will
            be unchanged. If a BoolTensor is provided, positions with True is
            not allowed to attend while False values will be unchanged. If a
            FloatTensor is provided, it will be added to the attention weight.
        pos_embs: torch.Tensor, optional
            Positional embeddings added to the attention map of shape (L, S, E) or (L, S, 1).

        Outputs
        -------
        attn_output : torch.Tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_output_weights : torch.Tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
        """
        # give tensors of shape (time, batch, fea)
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        # this will be legit because of https://github.com/pytorch/pytorch/blob/5288d05cfdda85c46c4df84617fa7f37c21b10b3/torch/nn/functional.py#L4946
        # we can inject relative learnable pos embeddings directly in MHA via the attn_mask
        if pos_embs is not None:
            if attn_mask is not None:
                attn_mask += pos_embs
            else:
                attn_mask = pos_embs

        output = self.att(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn_weights,
        )

        if return_attn_weights:
            output, attention_weights = output
            # reshape the output back to (batch, time, fea)
            output = output.permute(1, 0, 2)
            return output, attention_weights
        else:
            output = output.permute(1, 0, 2)
            return output


class PositionalwiseFeedForward(torch.nn.Module):
    """The class implements the positional-wise feed forward module in
    “Attention Is All You Need”.

    Arguments
    ----------
    d_ffn: int
        Hidden layer size.
    input_shape : tuple, optional
        Expected shape of the input. Alternatively use ``input_size``.
    input_size : int, optional
        Expected size of the input. Alternatively use ``input_shape``.
    dropout: float, optional
        Dropout rate.
    activation: torch.nn.Module, optional
        activation functions to be applied (Recommendation: ReLU, GELU).

    Example
    -------
    >>> inputs = torch.rand([8, 60, 512])
    >>> net = PositionalwiseFeedForward(256, input_size=inputs.shape[-1])
    >>> outputs = net(inputs)
    >>> outputs.shape
    torch.Size([8, 60, 512])
    """

    def __init__(
        self,
        d_ffn,
        input_shape=None,
        input_size=None,
        dropout=0.0,
        activation=torch.nn.ReLU,
    ):
        super().__init__()

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(input_size, d_ffn),
            activation(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_ffn, input_size),
        )

    def forward(self, x):
        """Applies PositionalwiseFeedForward to the input tensor x."""
        # give a tensor of shap (time, batch, fea)
        x = x.permute(1, 0, 2)
        x = self.ffn(x)

        # reshape the output back to (batch, time, fea)
        x = x.permute(1, 0, 2)

        return x


class RelPosMHAXL(torch.nn.Module):
    """ This class implements the relative multihead implementation similar to that in Transformer XL
    https://arxiv.org/pdf/1901.02860.pdf

    Arguments
    ---------
    embed_dim : int
        Size of the encoder feature vectors from which keys and values are computed.
    num_heads: int
        Number of attention heads.
    dropout : float, optional
        Dropout rate.
    vbias: bool, optional
        Whether to use bias for computing value.
    vdim: int, optional
        Size for value. Default is embed_dim (Note each head is embed_dim // num_heads).
    mask_pos_future: bool, optional
        Whether to mask future positional encodings values.
        Must be true for causal applications e.g. decoder.
    Example
    -------
    >>> inputs = torch.rand([6, 60, 512])
    >>> pos_emb = torch.rand([1, 2*60-1, 512])
    >>> net = RelPosMHAXL(num_heads=8, embed_dim=inputs.shape[-1])
    >>> outputs, attn = net(inputs, inputs, inputs, pos_emb)
    >>> outputs.shape
    torch.Size([6, 60, 512])
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        vbias=False,
        vdim=None,
        mask_pos_future=False,
    ):
        super(RelPosMHAXL, self).__init__()
        self.embed_dim = embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.vdim == embed_dim
        self.mask_pos_future = mask_pos_future
        self.vbias = vbias

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.vhead_dim = self.vdim // num_heads

        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        assert (
            self.vhead_dim * num_heads == self.vdim
        ), "vdim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.qk_proj_weight = torch.nn.Parameter(
                torch.empty(2 * embed_dim, embed_dim)
            )
            self.v_proj_weight = torch.nn.Parameter(torch.empty(self.vdim, embed_dim))
        else:
            self.in_proj_weight = torch.nn.Parameter(
                torch.empty(3 * embed_dim, embed_dim)
            )

        if vbias:
            self.value_bias_weight = torch.nn.Parameter(torch.empty(self.vdim))
        else:
            self.vbias = None

        self.dropout_att = torch.nn.Dropout(dropout)
        self.out_proj = torch.nn.Linear(self.vdim, embed_dim)

        self.linear_pos = torch.nn.Linear(embed_dim, embed_dim, bias=False)

        self.pos_bias_u = torch.nn.Parameter(
            torch.empty(self.head_dim, self.num_heads)
        )
        self.pos_bias_v = torch.nn.Parameter(
            torch.empty(self.head_dim, self.num_heads)
        )

        if next(self.parameters()).dtype == torch.float16:
            self.attn_fill_value = -65000
        else:
            self.attn_fill_value = -float("inf")

        self._reset_parameters()
        self.scale = 1 / math.sqrt(self.embed_dim)

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            torch.nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            torch.nn.init.xavier_uniform_(self.qk_proj_weight)
            torch.nn.init.xavier_uniform_(self.v_proj_weight)

        if self.vbias is not None:
            torch.nn.init.constant_(self.value_bias_weight, 0.0)

        # positional biases
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Relative shift implementation."""
        # batch, head, time1, 2*time1-1.

        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)

        if self.mask_pos_future:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x[..., : pos_len // 2 + 1]

    def forward(
        self,
        query,
        key,
        value,
        pos_embs,
        key_padding_mask=None,
        attn_mask=None,
        return_attn_weights=True,
    ):
        """
        Arguments
        ----------
        query : tensor
            (B, L, E) where L is the target sequence length,
            B is the batch size, E is the embedding dimension.
        key : tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        value : tensor
            (B, S, E) where S is the source sequence length,
            B is the batch size, E is the embedding dimension.
        pos_emb : tensor
            bidirectional sinusoidal positional embedding tensor (1, 2*S-1, E) where S is the max length between source and target sequence lengths,
            and E is the embedding dimension.
        key_padding_mask : tensor
            (B, S) where B is the batch size, S is the source sequence
            length. If a ByteTensor is provided, the non-zero positions will
            be ignored while the position with the zero positions will be
            unchanged. If a BoolTensor is provided, the positions with the
            value of True will be ignored while the position with the value
            of False will be unchanged.
        attn_mask : tensor
            2D mask (L, S) where L is the target sequence length, S is
            the source sequence length.
            3D mask (N*num_heads, L, S) where N is the batch
            size, L is the target sequence length, S is the source sequence
            length. attn_mask ensure that position i is allowed to attend the
            unmasked positions. If a ByteTensor is provided, the non-zero
            positions are not allowed to attend while the zero positions will
            be unchanged. If a BoolTensor is provided, positions with True is
            not allowed to attend while False values will be unchanged. If a
            FloatTensor is provided, it will be added to the attention weight.

        Outputs
        -------
        out : tensor
            (B, L, E) where L is the target sequence length, B is the
            batch size, E is the embedding dimension.
        attn_score : tensor
            (B, L, S) where B is the batch size, L is the target
            sequence length, S is the source sequence length.
        """

        # query, key and value are of shape batch, time, embed_dim
        bsz = query.shape[0]
        klen = key.shape[1]
        qlen = query.shape[1]

        if self._qkv_same_embed_dim:
            # self-attention
            if (query is key or torch.equal(query, key)) and (
                key is value or torch.equal(key, value)
            ):
                query, key, value = (
                    F.linear(query, self.in_proj_weight)
                    .view(bsz, -1, self.num_heads, self.head_dim * 3)
                    .chunk(3, dim=-1)
                )
            else:
                qweight, kweight, vweight = self.in_proj_weight.chunk(3, dim=0)
                query = F.linear(query, qweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
                key = F.linear(key, kweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
                value = F.linear(value, vweight).view(
                    bsz, -1, self.num_heads, self.head_dim
                )
        else:
            raise NotImplementedError
            query, key = (
                nn.functional.linear(query, self.qk_proj_weight)
                .view(bsz, -1, self.num_heads, self.head_dim * 2)
                .chunk(2, dim=-1)
            )
            value = nn.functional.linear(value, self.v_proj_weight).view(
                bsz, -1, self.num_heads, self.vhead_dim
            )

        if self.vbias is not None:
            value = value + self.value_bias_weight.view(
                1, 1, self.num_heads, self.vhead_dim
            )

        p_k = self.linear_pos(pos_embs).view(
            1, -1, self.num_heads, self.head_dim
        )
        # (batch, head, klen, d_k)

        q_with_bias_u = (
            query + self.pos_bias_u.view(1, 1, self.num_heads, self.head_dim)
        ).transpose(1, 2)
        # (batch, head, qlen, d_k)
        q_with_bias_v = (
            query + self.pos_bias_v.view(1, 1, self.num_heads, self.head_dim)
        ).transpose(1, 2)

        # (batch, head, qlen, klen)
        matrix_ac = torch.matmul(q_with_bias_u, key.permute(0, 2, 3, 1))
        # (batch, num_heads, klen, 2*klen-1)
        matrix_bd = torch.matmul(q_with_bias_v, p_k.permute(0, 2, 3, 1))
        matrix_bd = self.rel_shift(matrix_bd)  # shifting trick

        # if klen != qlen:
        #   import ipdb
        #  ipdb.set_trace(

        attn_score = (matrix_ac + matrix_bd) * self.scale

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.view(1, 1, qlen, klen)
            else:
                attn_mask = attn_mask.view(-1, self.num_heads, qlen, klen)

            if attn_mask.dtype == torch.bool:
                attn_score = attn_score.masked_fill(
                    attn_mask, self.attn_fill_value
                )
            else:
                attn_score += attn_mask

        if key_padding_mask is not None:
            attn_score = attn_score.masked_fill(
                key_padding_mask.view(bsz, 1, 1, klen), self.attn_fill_value,
            )

        attn_score = F.softmax(attn_score, dim=-1)
        attn_score = self.dropout_att(attn_score)
        x = torch.matmul(
            attn_score, value.transpose(1, 2)
        )  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(bsz, -1, self.vhead_dim * self.num_heads)
        )  # (batch, time1, d_model)

        out = self.out_proj(x)
        if return_attn_weights:
            return out, attn_score
        return out

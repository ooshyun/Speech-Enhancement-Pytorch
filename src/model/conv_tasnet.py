"""
https://github.com/kaituoxu/Conv-TasNet/blob/master/src/conv_tasnet.py
https://github.com/facebookresearch/demucs/blob/v2/demucs/tasnet.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def overlap_and_add(signal, frame_step):
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes,
                         device=signal.device).unfold(0, subframes_per_frame, subframe_step)
    frame = frame.long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class ConvTasNet(nn.Module):
    def __init__(self,  # default
                 sources,
                 N=128,
                 L=40,
                 B=128,
                 H=256,
                 P=3,
                 X=7,
                 R=2,
                ######## 
                #  N=128,
                #  L=16,
                #  B=128,
                #  H=256,
                #  P=3,
                #  X=7,
                #  R=3,
                ######## 
                #  N=256,
                #  L=16,
                #  B=128,
                #  H=256,
                #  P=3,
                #  X=8,
                #  R=3,
                ########
                #  N=256,
                #  L=20,
                #  B=256,
                #  H=512,
                #  P=3,
                #  X=8,
                #  R=4,
                ########
                #  N=512,
                #  L=16,
                #  B=128,
                #  H=512,
                #  P=3,
                #  X=8, 
                #  R=3,
                 audio_channels=2,
                 norm_type="gLN",
                 causal=False,
                 mask_nonlinear='relu',
                 sample_rate=44100,
                 segment_length=44100 * 2 * 4,
                 skip=False,
                 *args,
                 **kwargs,
                 ):
        """
        Args:
            sources: list of sources
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 * 1-conv block
            H: Number of channels in convolutional blocks   
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat - Number of dilation
            R: Number of repeats                             - Number of TCN
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask

        Best:
            N: 512
            L: 16
            B: 128
            H: 512
            Sc: 128 # skip connection channels
            P: 3
            X: 8
            R: 3
            norm_type: gLN    
            causal: False
            
        [TODO] Diagram - Notion

        """
        super(ConvTasNet, self).__init__()
        # Hyper-parameter
        self.sources = sources
        self.C = len(sources)
        self.N, self.L, self.B, self.H, self.P, self.X, self.R = N, L, B, H, P, X, R
        self.norm_type = norm_type
        self.causal = causal
        self.mask_nonlinear = mask_nonlinear
        self.audio_channels = audio_channels
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        # Components
        self.encoder = Encoder(L, N, audio_channels)
        self.separator = TemporalConvNet(
            N, B, H, P, X, R, self.C, norm_type, causal, mask_nonlinear, skip=skip)
        self.decoder = Decoder(N, L, audio_channels)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def valid_length(self, length):
        return length

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        mixture_w = self.encoder(mixture)
        est_mask = self.separator(mixture_w)
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, L, N, audio_channels):
        super(Encoder, self).__init__()
        # Hyper-parameter
        self.L, self.N = L, N
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(audio_channels, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, N, L, audio_channels):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.N, self.L = N, L
        self.audio_channels = audio_channels
        # Components
        self.basis_signals = nn.Linear(N, audio_channels * L, bias=False)

    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [M, N, K]
            est_mask: [M, C, N, K]
        Returns:
            est_source: [M, C, T]
        """
        # D = W * M
        source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # [M, C, N, K]
        source_w = torch.transpose(source_w, 2, 3)  # [M, C, K, N]
        # S = DV
        est_source = self.basis_signals(source_w)  # [M, C, K, ac * L]
        m, c, k, _ = est_source.size()
        est_source = est_source.view(m, c, k, self.audio_channels, -1).transpose(2, 3).contiguous()
        est_source = overlap_and_add(est_source, self.L // 2)  # M x C x ac x T
        return est_source


EPS = 1e-8

class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, C, norm_type="gLN", causal=False, out_activation='relu', skip=True):
        """
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 * 1-conv block,
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            out_activation: use which non-linear function to generate mask


        [TODO] Diagram - Notion

        """
        super(TemporalConvNet, self).__init__()
        # Hyper-parameter
        self.C = C
        self.out_activation = out_activation
        # Components
        # [M, N, K] -> [M, N, K]
        layer_norm = ChannelwiseLayerNorm(N)
        # [M, N, K] -> [M, B, K]
        bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        # [M, B, K] -> [M, B, K]
        repeats = []
        for _ in range(R):
            blocks = []
            for x in range(X):
                dilation = 2**x
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                blocks += [
                    TemporalBlock(B, # in channels
                                  H, # out channels
                                  P, # kernel size
                                  stride=1,
                                  padding=padding,
                                  dilation=dilation,
                                  norm_type=norm_type,
                                  causal=causal,
                                  skip=skip)
                ]
            repeats += [nn.Sequential(*blocks)]
        temporal_conv_net = nn.Sequential(*repeats)
        
        # [TODO] where is prelu before pointwise convolution? 

        # [M, B, K] -> [M, C*N, K]
        mask_conv1x1 = nn.Conv1d(B,     # in channels
                                C * N,  # out channels
                                1,      # kernel size
                                bias=False)
        
        # Put together
        self.skip = skip
        if skip:
            self.layer_norm = layer_norm
            self.bottleneck_conv1x1 = bottleneck_conv1x1
            self.temporal_conv_net = temporal_conv_net
            self.mask_conv1x1 = mask_conv1x1
        else:
            self.network = nn.Sequential(layer_norm, bottleneck_conv1x1, temporal_conv_net,
                                     mask_conv1x1)

    def forward(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        M, N, K = mixture_w.size()

        # [M, N, K] -> [M, C*N, K]
        if self.skip:
            x = self.layer_norm(mixture_w)
            x = self.bottleneck_conv1x1(x)
            skip = None
            _, skip = self.temporal_conv_net([x, skip])
            x = self.mask_conv1x1(skip)
        else:
            x = self.network(mixture_w)
        
        x = x.view(M, self.C, N, K)  # [M, C*N, K] -> [M, C, N, K]
        if self.out_activation == 'softmax':
            est_mask = F.softmax(x, dim=1)
        elif self.out_activation == 'relu':
            est_mask = F.relu(x)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask


class TemporalBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 norm_type="gLN",
                 causal=False,
                 skip=True):
        super(TemporalBlock, self).__init__()
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size, stride, padding,
                                        dilation, norm_type, causal, skip=skip)
        
        # Put together
        # self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)
        self.skip = skip
        
    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        if self.skip:
            residual = x[0]
            skip_out = x[1]
            out, skip = self.net(x[0])
            return out+residual, skip+skip_out if skip_out is not None else skip
        else:
            residual = x
            out = self.net(x)
            return out+residual
        # TODO: when P = 3 here works fine, but when P = 2 maybe need to pad?
        # return F.relu(out + residual) # look like w/o F.relu is better than w/ F.relu


class DepthwiseSeparableConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 norm_type="gLN",
                 causal=False,
                 skip=True):
        super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        depthwise_conv = nn.Conv1d(in_channels,
                                   in_channels,
                                   kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_channels,
                                   bias=False)
        if causal:
            chomp = Chomp1d(padding)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, in_channels)
        # [M, H, K] -> [M, B, K]
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        
        # Put together
        if causal:
            self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm)
        else:
            self.net = nn.Sequential(depthwise_conv, prelu, norm)
        
        if skip:
            self.skip_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.skip = skip
        
    def forward(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        x = self.net(x)
        if self.skip:
            return self.pointwise_conv(x), self.skip_conv(x)
        else:
            return self.pointwise_conv(x)


class Chomp1d(nn.Module):
    """To ensure the output length is the same as the input.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x: [M, H, Kpad]
        Returns:
            [M, H, K]
        """
        return x[:, :, :-self.chomp_size].contiguous()


def chose_norm(norm_type, channel_size):
    """The input of normlization will be (M, C, K), where M is batch size,
       C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    elif norm_type == "id":
        return nn.Identity()
    else:  # norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)


# TODO: Use nn.LayerNorm to impl cLN to speed up
class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""
    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y

if __name__ == "__main__":
    # First checking if GPU is available
    train_on_gpu=torch.cuda.is_available()

    def get_model():
        return ConvTasNet

    if(train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')
        
    device = torch.device('cuda' if train_on_gpu else 'cpu')

    torch.manual_seed(123)
    M, N, L, T = 2, 3, 4, 12
    K = 2 * T // L - 1
    B, H, P, X, R, C, norm_type, causal = 2, 3, 3, 3, 2, 2, "gLN", False
    mixture = torch.randn((M, T))
    encoder = Encoder(L, N, C)
    mixture_w = encoder(mixture[None])
    print('mixture', mixture.size())
    print('mixture_w', mixture_w.size())

    # test TemporalConvNet
    separator = TemporalConvNet(N, B, H, P, X, R, C, norm_type=norm_type, causal=causal)
    est_mask = separator(mixture_w)
    print('est_mask', est_mask.size())

    # test Decoder
    decoder = Decoder(N, L, C)
    est_source = decoder(mixture_w, est_mask)
    print('est_source', est_source.size)

    # test Conv-TasNet
    conv_tasnet = ConvTasNet([None], N, L, B, H, P, X, R, C, norm_type=norm_type)
    est_source = conv_tasnet(mixture[None])
    print('est_source size', est_source.size())


    # First checking if GPU is available
    train_on_gpu=torch.cuda.is_available()

    def get_model():
        return ConvTasNet

    if(train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')
        
    device = torch.device('cuda' if train_on_gpu else 'cpu')

    import argparse
    parser = argparse.ArgumentParser(
        "denoiser.demucs",
        description="Benchmark the streaming Demucs implementation, "
                    "as well as checking the delta with the offline implementation.")
    parser.add_argument("--num_sources", default=2, type=int)
    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument("--segment", default=1, type=float)
    parser.add_argument("--audio_channels", default=2, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--channels_interval", default=24, type=int)
    parser.add_argument("--skip", default=False, type=bool)    
    parser.add_argument("--device", default="cpu", type=str)
        
    args = parser.parse_args()
    
    model = get_model()([None]*args.num_sources, N, L, B, H, P, X, R, # C(audio_channels)
                        audio_channels=args.audio_channels,
                        sample_rate=args.sample_rate,
                        norm_type=norm_type,
                        skip=args.skip).to(args.device)

    length = int(args.sample_rate*args.segment) 
    x = torch.randn(args.audio_channels, length).to(args.device)
    print("Input: ", x[None].shape)
    out = model(x[None])
    print("Output: ", out.shape)
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2**20
    print(f"model size: {model_size:.5f}MB")


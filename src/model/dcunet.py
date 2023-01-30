"""
https://github.com/sweetcocoa/DeepComplexUNetPyTorch.git
https://github.com/pheepa/DCUnet
"""
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, complex=False, padding_mode="zeros"):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding
            
        if complex:
            conv = ComplexConv2d
            bn = ComplexBatchNorm2d
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d

        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0, 0), complex=False):
        super().__init__()
        if complex:
            tconv = ComplexConvTranspose2d
            bn = ComplexBatchNorm2d
        else:
            tconv = nn.ConvTranspose2d
            bn = nn.BatchNorm2d
        
        self.transconv = tconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DCUnet(nn.Module):
    def __init__(self, input_channels=1,
                 data_type=False,
                 model_complexity=45,
                 model_depth=20,
                 padding_mode="zeros",
                 *args,
                 **kwargs):
        super().__init__()

        if data_type:
            model_complexity = int(model_complexity // 1.414)

        self.set_size(model_complexity=model_complexity, input_channels=input_channels, model_depth=model_depth)
        
        self.amplitude = Amplitude()
        self.encoders = []
        self.model_length = model_depth // 2

        for i in range(self.model_length):
            module = Encoder(self.enc_channels[i], self.enc_channels[i + 1], kernel_size=self.enc_kernel_sizes[i],
                             stride=self.enc_strides[i], padding=self.enc_paddings[i], complex=data_type, padding_mode=padding_mode)
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

        self.decoders = []

        for i in range(self.model_length):
            module = Decoder(self.dec_channels[i] + self.enc_channels[self.model_length - i], self.dec_channels[i + 1], kernel_size=self.dec_kernel_sizes[i],
                             stride=self.dec_strides[i], padding=self.dec_paddings[i], complex=data_type)
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)

        if data_type:
            conv = ComplexConv2d
        else:
            conv = nn.Conv2d

        linear = conv(self.dec_channels[-1], 1, 1)

        self.add_module("linear", linear)
        self.data_type = data_type
        self.padding_mode = padding_mode

        self.decoders = nn.ModuleList(self.decoders)
        self.encoders = nn.ModuleList(self.encoders)

    def forward(self, x):
        x = x.transpose(2, 3) # channel, F, T, real/imag -> # channel, T, F, real/imag
        if self.data_type:
            x = x
        else:
            x = self.amplitude(x)

        # go down
        xs = []
        for i, encoder in enumerate(self.encoders):
            xs.append(x)
            #print("x{}".format(i), x.shape)
            x = encoder(x)
        # xs : x0 : input x1 ... x9

        #print(x.shape)
        p = x
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i == self.model_length - 1:
                break
            #print(f"p{i}, {p.shape} + x{self.model_length - 1 - i}, {xs[self.model_length - 1 -i].shape}, padding {self.dec_paddings[i]}")
            p = torch.cat([p, xs[self.model_length - 1 - i]], dim=1)

        #print(p.shape)
        mask = self.linear(p)
        mask = torch.tanh(mask)
        mask = mask.transpose(2, 3) # channel, T, F, real/imag -> # channel, F, T, real/imag
        return mask

    def set_size(self, model_complexity, model_depth=20, input_channels=1):
        if model_depth == 10:
            self.enc_channels = [input_channels,
                                 model_complexity,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 ]
            self.enc_kernel_sizes = [(7, 5),
                                     (7, 5),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3)]
            self.enc_strides = [(2, 2),
                                (2, 2),
                                (2, 2),
                                (2, 2),
                                (2, 1)]
            self.enc_paddings = [(2, 1),
                                 None,
                                 None,
                                 None,
                                 None]

            self.dec_channels = [0,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2]

            self.dec_kernel_sizes = [(4, 3),
                                     (4, 4),
                                     (6, 4),
                                     (6, 4),
                                     (7, 5)]

            self.dec_strides = [(2, 1),
                                (2, 2),
                                (2, 2),
                                (2, 2),
                                (2, 2)]

            self.dec_paddings = [(1, 1),
                                 (1, 1),
                                 (2, 1),
                                 (2, 1),
                                 (2, 1)]

        elif model_depth == 20:
            self.enc_channels = [input_channels,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 128]

            self.enc_kernel_sizes = [(7, 1),
                                     (1, 7),
                                     (6, 4),
                                     (7, 5),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3)]

            self.enc_strides = [(1, 1),
                                (1, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1)]

            self.enc_paddings = [(3, 0),
                                 (0, 3),
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None,
                                 None]

            self.dec_channels = [0,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2]

            self.dec_kernel_sizes = [(4, 3),
                                     (4, 2),
                                     (4, 3),
                                     (4, 2),
                                     (4, 3),
                                     (4, 2),
                                     (6, 3),
                                     (7, 5),
                                     (1, 7),
                                     (7, 1)]

            self.dec_strides = [(2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (1, 1),
                                (1, 1)]

            self.dec_paddings = [(1, 1),
                                 (1, 0),
                                 (1, 1),
                                 (1, 0),
                                 (1, 1),
                                 (1, 0),
                                 (2, 1),
                                 (2, 1),
                                 (0, 3),
                                 (3, 0)]
        else:
            raise ValueError("Unknown model depth : {}".format(model_depth))

class ComplexConv2d(nn.Module):
    # https://github.com/litcoderr/ComplexCNN/blob/master/complexcnn/modules.py
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias, **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__()

        ## Model components
        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation,
                                           **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps, track_running_stats=track_running_stats, **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output


class Amplitude(nn.Module):
    def __init__(self,
                *args,
                **kwarg):
        super().__init__()
    def forward(self, inputs):
        assert inputs.size()[-1] == 2, f"Tensor needs real and imag in the last rank..."
        return torch.abs(torch.pow(inputs[..., 0], exponent=2) - torch.pow(inputs[..., 1], exponent=2))


if __name__ == "__main__":
    # First checking if GPU is available
    train_on_gpu=torch.cuda.is_available()

    def get_model():
        return DCUnet

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
    parser.add_argument("--sample_rate", default=16000, type=int)
    parser.add_argument("--segment", default=1.024, type=float)
    parser.add_argument("--input_channels", default=1, type=int)
    parser.add_argument("--n_fft", default=512, type=int)
    parser.add_argument("--hop_length", default=256, type=int)
    parser.add_argument("--model_depth", default=10, type=int) # 10, 20
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--data_type", default=True, type=bool)
    parser.add_argument("--model_complexity", default=45, type=int) # 45, 90
    parser.add_argument("--padding_mode", default="zeros", type=str)

    args = parser.parse_args()
    
    model = get_model()(input_channels=args.input_channels, 
                        model_depth=args.model_depth, 
                        data_type=args.data_type, 
                        model_complexity=args.model_complexity, 
                        padding_mode=args.padding_mode).to(args.device)

    nframe = int(int(args.sample_rate*args.segment) // args.hop_length) + 1
    nfeature = int(args.n_fft//2)+1

    x = torch.randn(1, nfeature, nframe, 2).to(args.device) # channel, F, T, real/imag
    print("In: ", x.shape)
    out = model(x[None])
    print("Out: ", out.shape)
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2**20
    print(f"model size: {model_size:.1f}MB")
    
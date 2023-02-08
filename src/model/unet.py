"""
Code format from,
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, unet_channels, unet_layer=4, bilinear=False, *args, **kwargs):
        super(UNet, self).__init__()
        self.unet_channels = unet_channels
        self.bilinear = bilinear
                
        #      1 -> 16 -> 32 -> 64 -> 128 -> 256
        # 1 <- 2 <- 16 <- 32 <- 64 <- 128 <- 256
        channel_interval = 16
        assert unet_channels < channel_interval, f"channel can be under {channel_interval}"
        channel = [unet_channels] + [2**n*channel_interval for n in range(unet_layer+1)]

        encoder = []
        for n in range(unet_layer):
            encoder.append(Down(in_channels=channel[n], out_channels=channel[n+1], dropout=0. if n < unet_layer-1 else 0.5))
        self.encoder = nn.ModuleList(encoder)

        self.middle = DoubleConv(in_channels=channel[-2], out_channels=channel[-1], dropout=0.5)
        
        channel = channel[::-1]
        decoder = []
        for n in range(unet_layer):
            decoder.append(Up(in_channels=channel[n], 
                            out_channels=channel[n+1], 
                            bilinear=bilinear,
                            first=False if n > 0 else True))
        self.decoder = nn.ModuleList(decoder)

        self.outconv = Up(in_channels=channel[-2], out_channels=channel[-1], bilinear=bilinear, last=True)

    def forward(self, mix):
        amp = torch.abs(mix[..., 0]**2+mix[..., 1]**2)
        x = amp
        res = []
        # print("UP!")
        # print(x.shape, len(self.encoder))
        for encoder in self.encoder:
            x = encoder(x)
            res.append(x)
            # print(x.shape)
        
        # print("Middle!")
        x = self.middle(x)
        # print(x.shape)
            
        # print("Down!")
        for decoder in self.decoder:
            res_buf = res.pop()
            x = decoder(x, res_buf)
            # print(x.shape)
    
        # print("Last!")
        x = self.outconv(x, amp)
        x = mix * torch.unsqueeze(x, dim=-1)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0):
        super(DoubleConv, self).__init__()
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.mid_channels = mid_channels
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=False),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with double conv and then max pool"""

    def __init__(self, in_channels, out_channels, dropout=0):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels, dropout=dropout),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, first=False, last=False):
        super(Up, self).__init__()
        self.first = first
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            if not self.first:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels+out_channels, out_channels, in_channels // 2)
        else:
            if not self.first:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  
            
            if first:
                in_channels_conv = in_channels + out_channels  
            elif last:
                in_channels_conv = in_channels//2 + out_channels
            else:
                in_channels_conv = in_channels

            self.conv = DoubleConv(in_channels_conv, out_channels)


    def forward(self, x1, x2):
        if not self.first:
            x1 = self.up(x1)

        # input is Channel Height Width - [TODO] ?
        # print(x1.shape, x2.shape)
        assert x2.size()[2] >= x1.size()[2] and x2.size()[3] >= x1.size()[3]
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        print(x1.shape, x2.shape)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        print(x1.shape, x2.shape)
        
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)
        
if __name__ == "__main__":
    # First checking if GPU is available
    train_on_gpu=torch.cuda.is_available()

    def get_model():
        return UNet

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
    parser.add_argument("--unet_channels", default=6, type=int)
    parser.add_argument("--unet_layer", default=4, type=int)
    parser.add_argument("--n_fft", default=512, type=int)
    parser.add_argument("--window_length", default=512, type=int)
    parser.add_argument("--hop_length", default=256, type=int)
    parser.add_argument("--window_type", default="hann", type=str)
    parser.add_argument("--bilinear", default=True, type=bool)
    parser.add_argument("--device", default="cpu", type=str)

    args = parser.parse_args()
    
    model = get_model()(unet_channels=args.unet_channels,
                        unet_layer=args.unet_layer,
                        bilinear=args.bilinear).to(args.device)
    
    nframe = int(int(args.sample_rate*args.segment) // args.hop_length) + 1
    nfeature = int(args.n_fft//2)+1
    x = torch.randn(args.unet_channels, nfeature, nframe, 2).to(args.device) # channel, F, T, real/imag
    print(f"In: wav {x.shape}")

    out_wav = model(x[None])
    print(f"Out: wav {out_wav.shape}")

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2**20
    print(f"model size: {model_size:.1f}MB")
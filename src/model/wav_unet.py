"""https://github.com/haoxiangsnr/Wave-U-Net-for-Speech-Enhancement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, ipt):
        return self.main(ipt)

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, ipt):
        return self.main(ipt)

class WavUnet(nn.Module):
    def __init__(self, unet_nlayers=12, channels_interval=24, *args, **kwargs):
        super(WavUnet, self).__init__()

        self.n_layers = unet_nlayers
        self.channels_interval = channels_interval
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        #          1    => 2    => 3    => 4    => 5    => 6   => 7   => 8   => 9  => 10 => 11 =>12
        # 16384 => 8192 => 4096 => 2048 => 1024 => 512 => 256 => 128 => 64 => 32 => 16 =>  8 => 4
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input):
        tmp = []
        o = input

        # print("Up Sampling!")
        # Up Sampling
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            tmp.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]
            # print(tmp[i].shape, o.shape)

        # print("Middle!")
        o = self.middle(o)
        # print(o.shape)

        # print("Down Sampling!")
        # Down Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # print(o.shape, tmp[self.n_layers - i - 1].shape)
            # Skip Connection
            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            # print(o.shape)
            o = self.decoder[i](o)

        o = torch.cat([o, input], dim=1)
        # print(o.shape)
        o = self.out(o)
        return o

if __name__ == "__main__":
    # First checking if GPU is available
    train_on_gpu=torch.cuda.is_available()

    def get_model():
        return WavUnet

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
    parser.add_argument("--unet_nlayers", default=12, type=int)
    parser.add_argument("--channels_interval", default=24, type=int)
    parser.add_argument("--device", default="cpu", type=str)
        
    args = parser.parse_args()
    
    model = get_model()(unet_nlayers=args.unet_nlayers,
                        channels_interval=args.channels_interval).to(args.device)

    length = int(args.sample_rate*args.segment) 
    x = torch.randn(args.input_channels, length).to(args.device)
    print(f"Input: ", x[None].shape)

    out = model(x[None])
    print(f"Out: {out.shape}")
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2**20
    print(f"model size: {model_size:.1f}MB")
    
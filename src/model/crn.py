"""

"""
import torch
import torch.nn as nn

class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 1)
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 2),
            stride=(2, 1),
            output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class CRN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self, *args, **kwarg):
        super(CRN, self).__init__()
        # Encoder
        self.conv_block_1 = CausalConvBlock(1, 16)
        self.conv_block_2 = CausalConvBlock(16, 32)
        self.conv_block_3 = CausalConvBlock(32, 64)
        self.conv_block_4 = CausalConvBlock(64, 128)
        self.conv_block_5 = CausalConvBlock(128, 256)

        # LSTM
        # n_channels * n_f_bins
        self.lstm_layer = nn.LSTM(input_size=1792, hidden_size=1792, num_layers=2, batch_first=True)

        self.tran_conv_block_1 = CausalTransConvBlock(256 + 256, 128)
        self.tran_conv_block_2 = CausalTransConvBlock(128 + 128, 64)
        self.tran_conv_block_3 = CausalTransConvBlock(64 + 64, 32)
        self.tran_conv_block_4 = CausalTransConvBlock(32 + 32, 16, output_padding=(1, 0))
        self.tran_conv_block_5 = CausalTransConvBlock(16 + 16, 1, is_last=True)

    def forward(self, x):
        self.lstm_layer.flatten_parameters() # ?
        amplitude = torch.sqrt(torch.pow(x[..., 0], 2.) - torch.pow(x[..., 1], 2.))

        # print(amplitude.shape)
        e_1 = self.conv_block_1(amplitude)
        # print(e_1.shape)
        e_2 = self.conv_block_2(e_1)
        # print(e_2.shape)
        e_3 = self.conv_block_3(e_2)
        # print(e_3.shape)
        e_4 = self.conv_block_4(e_3)
        # print(e_4.shape)
        e_5 = self.conv_block_5(e_4)  # [2, 256, 4, 200]
        # print(e_5.shape)

        batch_size, n_channels, n_f_bins, n_frame_size = e_5.shape

        # n_fft=512, [2, 256, 7, 200] = [2, (n_f_bins*n_channels) 1792, 200] => [2, 200, 1792]

        # [2, 256, 4, 200] = [2, 1024, 200] => [2, 200, 1024]
        lstm_in = e_5.reshape(batch_size, n_channels * n_f_bins, n_frame_size).permute(0, 2, 1)
        # print(lstm_in.shape)
        
        lstm_out, _ = self.lstm_layer(lstm_in)  # [2, 200, 1024]
        # print(lstm_out.shape)
        lstm_out = lstm_out.permute(0, 2, 1).reshape(batch_size, n_channels, n_f_bins, n_frame_size)  # [2, 256, 4, 200]
        # print(lstm_out.shape)

        d_1 = self.tran_conv_block_1(torch.cat((lstm_out, e_5), 1))
        # print(d_1.shape)
        d_2 = self.tran_conv_block_2(torch.cat((d_1, e_4), 1))
        # print(d_2.shape)
        d_3 = self.tran_conv_block_3(torch.cat((d_2, e_3), 1))
        # print(d_3.shape)
        d_4 = self.tran_conv_block_4(torch.cat((d_3, e_2), 1))
        # print(d_4.shape)
        d_5 = self.tran_conv_block_5(torch.cat((d_4, e_1), 1))
        # print(d_5.shape, x.shape)
        
        d_5 = torch.unsqueeze(input=d_5, dim=-1)
        # print(d_5.shape, x.shape)
        out = d_5*x
        
        return out


if __name__ == '__main__':
    # First checking if GPU is available
    train_on_gpu=torch.cuda.is_available()

    def get_model():
        return CRN

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
    parser.add_argument("--n_fft", default=512, type=int)
    parser.add_argument("--hop_length", default=256, type=int)
    parser.add_argument("--device", default="cpu", type=str)

    args = parser.parse_args()
    
    model = get_model()().to(args.device)

    length = int(args.sample_rate*args.segment) 
    nframe = int(int(args.sample_rate*args.segment) // args.hop_length) + 1
    nfeature = int(args.n_fft//2)+1
    
    # nfeature = 161
    # n_fft = 2*(nfeature-1)
    # print("NFFT ", n_fft)

    # [B, C, F, T]
    x = torch.randn(1, nfeature, nframe).to(args.device)
    print("In: ",x[None].shape)
    out = model(x[None])
    print("Out: ", out.shape)
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2**20
    print(f"model size: {model_size:.1f}MB")
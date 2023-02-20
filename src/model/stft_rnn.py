import numpy as np
import torch
import torch.nn as nn

class RNNBaseSTFTMask(nn.Module):
    def __init__(self, 
                num_spk=2,
                audio_channels=2,
                n_fft=512, 
                hop_length=256,
                sample_rate=16000, 
                rnn_hidden=256,
                rnn_layer=2,
                rnn_type="rnn",
                drop_out=0.5,
                activation="relu",
                bidirectional=False,
                *args,
                **kwarg):
        super(RNNBaseSTFTMask, self).__init__()

        self.audio_channels = audio_channels
        self.num_spk = num_spk
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        n_features = n_fft//2+1
        self.ampltude = Amplitude()
        self.phase = Phase()
        
        # torch.rnn batch_first
        # https://discuss.pytorch.org/t/could-someone-explain-batch-first-true-in-lstm/15402/9
        # (batch, seq, feature)
        # Without batch_first=True it will use the first dimension as the sequence dimension.
        # With batch_first=True it will use the second dimension as the sequence dimension.
        # out[-1]    # If batch_first=True OR
        # out[:, -1] # If batch_dirst=False
        if rnn_type == "rnn":
            self.rnn = nn.RNN(input_size=n_features, 
                        hidden_size=rnn_hidden, 
                        num_layers=rnn_layer, 
                        bias=False, 
                        dropout=drop_out,
                        batch_first=False,
                        bidirectional=bidirectional)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=n_features, 
                        hidden_size=rnn_hidden, 
                        num_layers=rnn_layer, 
                        bias=False, 
                        dropout=drop_out,
                        batch_first=False, 
                        bidirectional=bidirectional)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(input_size=n_features, 
                        hidden_size=rnn_hidden, 
                        num_layers=rnn_layer, 
                        bias=False, 
                        dropout=drop_out,
                        batch_first=False, 
                        bidirectional=bidirectional)
        
        self.batchnorm = nn.BatchNorm1d(num_features=rnn_hidden if not bidirectional else rnn_hidden*2)
        
        linear = nn.Linear(in_features=rnn_hidden if not bidirectional else rnn_hidden*2, 
                                out_features=n_features*num_spk, 
                                bias=True)
        if activation=="relu":
            activation = nn.ReLU()
        
        self.fc_layers = nn.Sequential(
            linear,
            activation,
        )

    def forward(self, inputs):
        # print(inputs.shape)
        mask = self.ampltude(inputs)
        batch, nchannel, nfeature, nframe = mask.shape 
        # batch, features, seq
        mask = torch.reshape(mask, shape=(batch*nchannel, nfeature, nframe)) # merge channel
        # print(mask.shape)
        mask = torch.transpose(mask, 1, 2)
        # print(mask.shape)
        mask, _ = self.rnn(mask) # batch, seq, features
        # print(mask.shape)
        mask = torch.transpose(mask, 1, 2)
        # print(mask.shape)
        mask = self.batchnorm(mask) # batch, features, seq
        # print(mask.shape)
        mask = torch.transpose(mask, 1, 2)
        # print(mask.shape)
        mask = self.fc_layers(mask) # batch, seq, features
        # print(mask.shape)
        mask = torch.transpose(mask, -1, -2) # batch, seq, features 
        # print(mask.shape)
        batch, nfeature, nframe = mask.shape
        mask = torch.reshape(mask, shape=(batch, self.num_spk, int(nfeature//self.num_spk), nframe))
        # mask = mask.view(batch, self.num_spk, int(nfeature//self.num_spk), nframe)
        # print(mask.shape)
        mask = torch.reshape(mask, shape=(batch//nchannel, nchannel, self.num_spk, int(nfeature//self.num_spk), nframe))
        # mask = mask.view(batch//nchannel, nchannel, self.num_spk, int(nfeature//self.num_spk), nframe)
        # print(mask.shape)
        mask = torch.transpose(mask, 1, 2)
        # print(mask.shape)

        mask = torch.unsqueeze(mask, dim=-1) # dtype expand
        out = mask*torch.unsqueeze(inputs, dim=1)
        return out

class Amplitude(nn.Module):
    def __init__(self,
                *args,
                **kwarg):
        super(Amplitude, self).__init__()
    def forward(self, inputs):
        assert inputs.size()[-1] == 2, f"Tensor needs real and imag in the last rank..."
        return torch.abs(torch.pow(inputs[..., 0], exponent=2) - torch.pow(inputs[..., 1], exponent=2))

class Phase(nn.Module):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eps = torch.tensor(np.ones(1, dtype=np.float32)*1e-5, dtype=torch.float32)
        
    def call(self, inputs, training=True):
        inputs[..., 1] += self.eps
        outputs = torch.angle(torch.complex(real=inputs[..., 0], imag=inputs[..., 1]))
        return outputs        

if __name__ == "__main__":
    # First checking if GPU is available
    train_on_gpu=torch.cuda.is_available()

    def get_model():
        return RNNBaseSTFTMask

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
    parser.add_argument("--audio_channels", default=2, type=int)
    parser.add_argument("--num_spk", default=2, type=int)
    parser.add_argument("--n_fft", default=512, type=int)
    parser.add_argument("--hop_length", default=256, type=int)
        
    parser.add_argument("--rnn_hidden", default=896, type=int)
    parser.add_argument("--rnn_layer", default=3, type=int)
    parser.add_argument("--rnn_type", default="lstm", type=str)
    parser.add_argument("--activation", default="relu", type=str)
    parser.add_argument("--bidirectional", default=True, type=bool)
    parser.add_argument("--drop_out", default=0.5, type=float)
    parser.add_argument("--device", default="cpu", type=str)

    args = parser.parse_args()
    
    model = get_model()(num_spk=args.num_spk,
                    audio_channels=args.audio_channels,
                    n_fft=args.n_fft, 
                    hop_length=args.hop_length,
                    sample_rate=args.sample_rate, 
                    rnn_hidden=args.rnn_hidden,
                    rnn_layer=args.rnn_layer,
                    rnn_type=args.rnn_type,
                    drop_out=args.drop_out,
                    activation=args.activation,
                    bidirectional=args.bidirectional,
                    ).to(args.device)

    nframe = int(int(args.sample_rate*args.segment) // args.hop_length) + 1
    nfeature = int(args.n_fft//2)+1

    x = torch.randn(args.audio_channels, nfeature, nframe, 2).to(args.device) # channel, T, F, real/imag
    out = model(x[None])[0]
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2**20
    print(f"model size: {model_size:.1f}MB")
    
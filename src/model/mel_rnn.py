"""
https://github.com/ooshyun/Speech-Enhancement-ML
"""
import torch
import torchaudio.transforms as transforms
import torch.nn as nn

class MelRNN(nn.Module):
    def __init__(self, 
                n_fft=512, 
                hop_length=256,
                n_mels=128,
                f_min=100,
                f_max=8000,
                sample_rate=16000, 
                rnn_hidden=256,
                rnn_layer=2,
                rnn_type="rnn",
                *args,
                **kwarg):
        super(MelRNN, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        
        self.amplitude = Amplitude()

        n_features = n_mels if n_mels else n_fft//2+1

        if n_mels:
            self.mel = transforms.MelScale(n_mels=n_mels,
                                                    sample_rate=sample_rate, 
                                                    f_min=f_min,
                                                    f_max=f_max,
                                                    n_stft=int(n_fft//2)+1,)

        if rnn_type == "rnn":
            self.rnn = nn.RNN(input_size=n_features, 
                        hidden_size=rnn_hidden, 
                        num_layers=rnn_layer, 
                        bias=False, 
                        batch_first=True, # True, (batch, seq, feature)
                        bidirectional=False)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=n_features, 
                        hidden_size=rnn_hidden, 
                        num_layers=rnn_layer, 
                        bias=False, 
                        batch_first=True, 
                        bidirectional=False)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(input_size=n_features, 
                        hidden_size=rnn_hidden, 
                        num_layers=rnn_layer, 
                        bias=False, 
                        batch_first=True, 
                        bidirectional=False)
        
        self.batchnorm = nn.BatchNorm1d(num_features=rnn_hidden)
        
        linear1 = nn.Linear(in_features=rnn_hidden, 
                                out_features=n_features, 
                                bias=True )
        activtaion1 = nn.ReLU()
        linear2 = nn.Linear(in_features=n_features,
                                out_features=n_features, 
                                bias=True )
        activtaion2 = nn.Sigmoid()
        

        self.fc_layers = nn.ModuleList([
            linear1,
            activtaion1,
            linear2,
            activtaion2,
        ])
        
        if n_mels:
            self.inverse_mel = transforms.InverseMelScale(n_stft=int(n_fft//2)+1, 
                                                                    n_mels=n_mels,
                                                                    sample_rate=sample_rate,
                                                                    f_min=f_min,
                                                                    f_max=f_max,
                                                                    max_iter=0,)

    def forward(self, inputs):
        x = self.amplitude(inputs)
        x = torch.squeeze(x, dim=1) # merge channel
        
        if self.n_mels:
            x = torch.pow(x, exponent=0.3)
            x = self.mel(x) 
        
        x = x.transpose(-1, -2)
        x, hn = self.rnn(x) # batch, seq, features
        x = x.transpose(-1, -2)
        
        x = self.batchnorm(x) # batch, features, seq
        
        x = x.transpose(-1, -2)        
        for fc_layer in self.fc_layers:
            x = fc_layer(x) # batch, features, seq 
        x = x.transpose(-1, -2)        

        if self.n_mels:
            x = self.inverse_mel(x)

        x = torch.unsqueeze(x, dim=1)  # unmerge channel
        x = inputs*torch.unsqueeze(x, dim=-1) # real, imag
        return x

class Amplitude(nn.Module):
    def __init__(self,
                *args,
                **kwarg):
        super(Amplitude, self).__init__()
    def forward(self, inputs):
        assert inputs.size()[-1] == 2, f"Tensor needs real and imag in the last rank..."
        return torch.abs(torch.pow(inputs[..., 0], exponent=2) - torch.pow(inputs[..., 1], exponent=2))

class MergeChannel(nn.Module):
    def __init__(self,
                *args,
                **kwarg):
        super(MergeChannel, self).__init__()
    def forward(self, inputs, channel=1, merge: bool=True):
        if merge:
            assert inputs.size()[1] == 1, f"MergeChannel supports mono channel"
            return torch.squeeze(inputs, dim=1)
        else:
            return torch.unsqueeze(inputs, dim=1)
        

if __name__ == "__main__":
    # First checking if GPU is available
    train_on_gpu=torch.cuda.is_available()

    def get_model():
        return MelRNN

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
    parser.add_argument("--n_fft", default=2048, type=int)
    parser.add_argument("--hop_length", default=1024, type=int)
    parser.add_argument("--n_mels", default=128, type=int)
    parser.add_argument("--f_min", default=125, type=int)
    parser.add_argument("--f_max", default=8000, type=int)
    parser.add_argument("--rnn_hidden", default=128, type=int)
    parser.add_argument("--rnn_layer", default=2, type=int)
    parser.add_argument("--rnn_type", default="rnn", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    
    args = parser.parse_args()
    
    model = get_model()(n_fft=args.n_fft,
                        hop_length=args.hop_length,
                        n_mels=args.n_mels,
                        f_min=args.f_min,
                        f_max=args.f_max,
                        sample_rate=args.sample_rate, 
                        rnn_hidden=args.rnn_hidden,
                        rnn_layer=args.rnn_layer,
                        rnn_type=args.rnn_type).to(args.device)

    nframe = int(int(args.sample_rate*args.segment) // args.hop_length) + 1
    nfeature = int(args.n_fft//2)+1

    x = torch.randn(1, nfeature, nframe, 2).to(args.device) # channel, T, F, real/imag
    out = model(x[None])[0]
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2**20
    print(f"model size: {model_size:.1f}MB")
    
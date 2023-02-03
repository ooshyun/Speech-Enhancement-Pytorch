"""
https://github.com/eesungkim/Speech_Enhancement_DNN_NMF.git
"""
import torch
import torch.nn as nn
from .ema import ExponentialMovingAverage
class NeuralNetwork(nn.Module):
    def __init__(self, 
                first=False,
                last=False,
                nfft=512, 
                hidden_layer=1024,
                bias=True,
                activation="leaky-relu",
                drop_out=0,
                *args,
                **kwargs):
        super().__init__()

        model = []
        nfeature = int(nfft//2+1)
        if first:
            model.append(nn.Linear(in_features=nfeature, out_features=hidden_layer, bias=bias))
            model.append(nn.BatchNorm1d(num_features=hidden_layer))
        elif last:
            model.append(nn.Linear(in_features=hidden_layer, out_features=nfeature, bias=bias))
            model.append(nn.BatchNorm1d(num_features=nfeature))
        else:
            model.append(nn.Linear(in_features=hidden_layer, out_features=hidden_layer, bias=bias))
            model.append(nn.BatchNorm1d(num_features=hidden_layer))
        
        if last:
            pass
        else:
            if activation == "linear":
                pass
            elif activation == "leaky-relu":
                model.append(nn.LeakyReLU(negative_slope=0.1))
            elif activation == "relu":
                model.append(nn.ReLU())
            elif activation == "sigmoid":
                model.append(nn.Sigmoid())
            elif activation == "tanh":
                model.append(nn.Tanh())
            else:
                raise ValueError(f"There is no implmentation for {activation}")
            model.append(nn.Dropout(p=drop_out))
        
        self.model = nn.ModuleList(model)
        
    def forward(self, mix:torch.Tensor):
        x = mix
        for layer in self.model:
            # print(x.shape, layer)   
            # dim = batch, n_feature, n_frame         
            # if isinstance(layer, nn.BatchNorm1d):
            #     batch, nframe, nfeature = x.shape
            #     x = x.reshape(batch, nfeature, nframe)
            x = layer(x)

            # if isinstance(layer, nn.BatchNorm1d):
            #     x = x.reshape(batch, nframe, nfeature)
        return x

class DeepNeuralNetwork(nn.Module):
    def __init__(self, 
                n_layer=4,
                dnn_method="mask",
                *args,
                **kwargs):
        super().__init__()

        model = []
        for n in range(n_layer):
            if n==0:
                model.append(NeuralNetwork(first=True, *args, **kwargs))
            elif n==n_layer-1:
                model.append(NeuralNetwork(last=True, *args, **kwargs))
            else:
                model.append(NeuralNetwork(*args, **kwargs))
            
        self.model = nn.ModuleList(model)
        self.dnn_method = dnn_method

        print(kwargs)
        if kwargs['dnn_ema']:
            self.ema = True
            n_feature = kwargs['n_fft']//2+1
            self.context = nn.Linear(in_features=n_feature, out_features=n_feature, bias=True)
            self.ema_in = ExponentialMovingAverage(alpha=0.1)
            self.ema_out = ExponentialMovingAverage(alpha=0.85)
        else:
            self.ema = False        

    def forward(self, mix:torch.Tensor):
        # [batch, channel, nfeature, nframe, ndtype] -> [batch * channel, nfeature, nframe]
        batch, n_channel, n_feature, n_frame, ndtype = mix.shape
        x = torch.sqrt(mix[..., 0]**2+mix[..., 1]**2)
        
        if n_channel == 1:
            x = x.squeeze()
        else:
            x = x.reshape(shape=(batch*n_channel, n_feature, n_frame))
    
        x = x.transpose(1, 2)
        
        if self.ema:
            x = self.context(x)
            x = self.ema_in(x)

        x = x.reshape(batch*n_channel*n_frame, n_feature)
        for layer in self.model:
            x = layer(x)
        x = x.reshape(batch*n_channel, n_frame, n_feature)

        if self.ema:
            x = self.ema_out(x)
            
        x = x.transpose(1, 2)

        # spectral estimation
        if self.dnn_method == "reconstruct":
            angle = torch.angle(x)
            zero_buffer = torch.zeros_like(x, dtype=x.dtype, device=x.device)
            x = x.type(torch.complex64) * torch.exp(torch.complex(real=zero_buffer, imag=angle))
            
            if n_channel == 1:
                x = x.unsqueeze(dim=1)
            else:
                x = x.reshape(shape=(batch, n_channel, n_feature, n_frame))
            x = torch.stack([torch.real(x), torch.imag(x)], dim=-1)

        # masking
        if self.dnn_method == "mask":
            if n_channel == 1:
                x = x.unsqueeze(dim=1)
            else:
                x = x.reshape(shape=(batch, n_channel, n_feature, n_frame))
            x = x.unsqueeze(dim=-1) # expand dtype
            x = mix*x

        return x
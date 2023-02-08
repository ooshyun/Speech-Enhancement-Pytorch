"""
Dlitated TCN, ED-TCN 
    - Paper reference. https://github.com/naplab/Conv-TasNet
    https://github.com/colincsl/TemporalConvolutionalNetworks
    https://github.com/kaituoxu/Conv-TasNet/blob/master/src/conv_tasnet.py
    https://github.com/asteroid-team/asteroid/blob/9dcf6ba3259ef2ffcc4a251f55d488b495644095/egs/whamr/TasNet/model.py#L32
    https://github.com/facebookresearch/demucs/blob/v2/demucs/tasnet.py
"""
import torch
import torch.nn as nn

class DilatedTCN(nn.Module):
    ...

class EncoderDecoderTCN(nn.Module):
    ...
    
class TFCN(nn.Module):
    """
    https://arxiv.org/pdf/2201.00480.pdf
    or https://web.cse.ohio-state.edu/~wang.77/papers/Pandey-Wang1.icassp19.pdf
    """
    def __init__(self):
        super(TFCN, self).__init__()
        ...
    
    def forward(self, mix):
        x = mix
        ...
        return x


if __name__ == "__main__":
    # First checking if GPU is available
    train_on_gpu=torch.cuda.is_available()

    def get_model():
        # return DilatedTCN
        return EncoderDecoderTCN

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
    parser.add_argument("--input_channels", default=2, type=int)
    parser.add_argument("--num_inputs", default=2, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--skip", default=True, type=bool)
        
    args = parser.parse_args()
    model = get_model()(num_inputs=args.num_inputs, 
                        num_channels=args.input_channels, ).to(args.device)

    length = int(args.sample_rate*args.segment) 
    x = torch.randn(args.num_inputs, length).to(args.device)
    print(f"Input: ", x[None].shape)

    out = model(x[None])
    print(f"Out: {out.shape}")
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2**20
    print(f"model size: {model_size:.1f}MB")
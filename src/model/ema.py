import torch
import torch.nn as nn

class ExponentialMovingAverage(nn.Module):
    """
        [B, T, ..., C]
        outputs_{t} = (1-alpha) * outputs_{t-1} + alpha * inputs_{t}
    """
    def __init__(
        self,
        alpha,
        *args,
        **kwargs,
    ):
        super(ExponentialMovingAverage, self).__init__()
        self.alpha = alpha
        
        ema0 = nn.Parameter(data=torch.ones((1, ), dtype=torch.float32)*alpha, requires_grad=False)
        ema1 = nn.Parameter(data=torch.ones((1, ), dtype=torch.float32)*(1-alpha), requires_grad=False)
    
        self.register_buffer('ema0', ema0)
        self.register_buffer('ema1', ema1)
    
    def forward(self, inputs):
        assert len(inputs.shape)==3
        _, time, _ = inputs.shape
        x = []

        delay_buffer = None
        for curr_time in range(time):
            if curr_time == 0:
                x.append(self.ema0 * inputs[:, curr_time, ...])
                delay_buffer = self.ema0 * inputs[:, curr_time, ...]
            elif curr_time == time-1:
                x.append(self.ema1*delay_buffer + self.ema0*inputs[:, curr_time, ...])
            else:
                x.append(self.ema1*delay_buffer + self.ema0*inputs[:, curr_time, ...])
                delay_buffer = self.ema1*delay_buffer + self.ema0*inputs[:, curr_time, ...]
        x = torch.stack(x, axis=1)
        return x

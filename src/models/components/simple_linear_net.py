from torch import nn
from einops.layers.torch import Rearrange

class SimpleLinearNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            Rearrange('b f -> b f'),
            nn.Linear(input_size, output_size, bias=True)
        )
    
    def forward(self, x):
        return self.net(x)
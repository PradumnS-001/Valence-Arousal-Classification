import torch
from torch import nn

class DSEAP1d(nn.Module):
    
    def __init__(self, in_channels:int, out_channels:int, kernel_sizes:list[int]=[7,5,5,3]):
        super().__init__()
        
        self.dconv = []
        for kernel in kernel_sizes:
            self.dconv.append(nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel,padding=kernel//2, groups=in_channels))
            self.dconv.append(nn.GELU())
        self.dconv.pop(-1)
        self.dconv = nn.Sequential(*self.dconv)
        
        self.bottleneck = nn.Sequential(
            nn.Linear(in_channels, max(1,in_channels//4)),
            nn.GELU(),
            nn.Linear(max(1,in_channels//4), in_channels),
            nn.Sigmoid()
        )
        
        self.out = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=1)
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        
        x = self.dconv(x)
        y = self.bottleneck(x.mean(dim=-1)).unsqueeze(-1)
        return self.out(x * y)
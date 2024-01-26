import torch
from torch import nn



class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: nn.Module = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.block(x)
        if self.shortcut:
            res = self.shortcut(res)
        x = x + res
        return x
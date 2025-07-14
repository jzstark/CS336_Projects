import torch 
from torch import nn
import numpy as np

from einops import rearrange, einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 device : torch.device | None = None, dtype : torch.dtype | None =None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32

        tensor = torch.empty(out_features, in_features, device=device, dtype=dtype)
        sigma = np.sqrt(2/(in_features + out_features))
        self.weight = torch.nn.init.trunc_normal_(tensor, \
            mean = 0.0, std = sigma, a = -3*sigma, b = 3*sigma)
        self.parameter = nn.Parameter(self.weight, requires_grad=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.parameter, x,
            'self_out_features self_in_features, ... self_in_features -> ... self_out_features')



    

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

    
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings # size of vocabulary
        self.embedding_dim = embedding_dim   # dim of each embedding vector, or d_model 
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32

        tensor = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.weight = torch.nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-3.0, b=3.0)
        self.parameter = nn.Parameter(self.weight, requires_grad=True)
    

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            token_ids (torch.Tensor): Tensor of shape (..., seq_len) containing token indices.
        
        Returns:
            torch.Tensor: Tensor of shape (..., seq_len, embedding_dim) containing embeddings.
        """
        return self.parameter[token_ids]


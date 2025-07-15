import torch 
from torch import nn
import numpy as np
from jaxtyping import Float, Int
from torch import Tensor

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



class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, 
                 device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype), requires_grad=True)
    
    def _rms(self, x: Float[Tensor, "self_d_model"]) -> Float:
        return torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

    def forward(self, x : Float[Tensor, " ... self_d_model"]) -> torch.Tensor:
        # input shape: (batch, seq_len, d_model) 
        in_dtype = x.dtype
        x = x.to(self.dtype)
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / norm * self.gain
        return x.to(in_dtype)
    

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, 
                 device: torch.device | None = None, 
                 dtype : torch.dtype  | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device if device is not None else torch.device('cpu')
        self.dtype  = dtype if dtype is not None else torch.float32

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model,    device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> torch.Tensor:
        x1 = torch.nn.functional.silu(self.w1(x))
        x2 = self.w3(x)
        return self.w2(x1 * x2)


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, 
                 device : torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device('cpu')

        assert d_k % 2 == 0, "d_k must be even for Rotary Positional Embedding, I guess"

        power  = torch.arange(1, d_k // 2 + 1, device=self.device) * 2 // d_k 
        theta_pow = torch.pow(self.theta, (-1) * power)
        
        #TODO: could be very large tables 
        
        """
        cosine = torch.zeros(max_seq_len, d_k // 2, device=self.device)
        sine   = torch.zeros(max_seq_len, d_k // 2, device=self.device)
        for i in range(max_seq_len):
            cosine[i,:] = torch.cos(i * theta_pow)
            sine[i,:] = torch.sin(i * theta_pow)
        
        self.register_buffer('cosine', cosine)
        self.register_buffer('sine', sine)
        """
        inv_freq : Float[Tensor, " d_k"] = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=self.device).float() / d_k))
        t : Float[Tensor, " max_seq_len"] = torch.arange(max_seq_len, device=self.device).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('cosine', torch.cos(freqs), persistent=False)
        self.register_buffer('sine', torch.sin(freqs), persistent=False)


    def forward(self, x: Float[Tensor, " ...  seq_len d_k"], 
                token_positions: Int[Tensor, "... seq_len"]) -> \
                Float[Tensor, "... seq_len d_k"]:
        input_shape = x.shape
        seq_len = input_shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}.")       

        """ Wrong and inefficient implementation

        result = torch.zeros_like(x, device=self.device, dtype=x.dtype)

        #I can use torch.einsum('aij, ijk -> aik', A, R) if a big R matrix is built
        for i in range(seq_len):
            pos : int = int(token_positions[..., i].item())
            #TODO: update building this large matrix every time 
            R = torch.zeros((self.d_k, self.d_k), device=self.device, dtype=x.dtype)
            for j in range(self.d_k - 1):
                R[j, j] = self.cosine[pos, j // 2] 
                R[j, j + 1] = - self.sine[pos, j // 2] 
                R[j + 1, j] = self.sine[pos, j // 2]
            
            result[..., i, :] = einsum(R, x[..., i, :],
                                  'self_d_k self_d_k, ... self_d_k -> ...   self_d_k')  
            #torch.matmul(x[..., i, :], R)
             
            tmp_x = torch.zeros(self.d_k, device=self.device, dtype=x.dtype)
            pos : int = int(token_positions[..., i].item())
            for j in range(self.d_k):
                tmp = torch.zeros(self.d_k, device=self.device, dtype=x.dtype)
                # TODO: the assumption is that d_k is even
                if j % 2 == 0:
                    tmp[j] = self.cosine[pos, j // 2]
                    tmp[j + 1] = self.sine[pos, j // 2]
                else:
                    tmp[j] = - self.sine[pos, j // 2]
                    tmp[j + 1] = self.cosine[pos, j // 2]
                
                tmp_x[j] = x[..., i, :] * tmp 
            x[..., i, :] = tmp_x.clone()
        return result
        """

        cos = self.cosine[..., token_positions, :]  # (..., seq_len, d_k//2)
        sin = self.sine[..., token_positions, :]    # (..., seq_len, d_k//2)
        # Split last dim into pairs
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # Apply rotation
        x_rotated_even = x1 * cos - x2 * sin
        x_rotated_odd  = x1 * sin + x2 * cos
        # Interleave even and odd back together
        x_rotated = torch.stack((x_rotated_even, x_rotated_odd), dim=-1)
        x_rotated = x_rotated.flatten(-2)
        return x_rotated
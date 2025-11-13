import torch
from einops import einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features:int, device=None, dtype=None):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(
            (out_features, in_features), 
            device=device, 
            dtype=dtype)
        )
        std = torch.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            self.W,
            mean=0,
            std=std,
            a=-3,
            b=3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        result = einsum(self.W, x, "out in, ... in -> ... out")
        return result

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(
            (num_embeddings, embedding_dim),
            device=device,
            dtype=dtype)
        )
        torch.nn.init.trunc_normal_(
            self.W,
            mean=0,
            std=1,
            a=-3,
            b=3
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor :
        return self.W[token_ids]
    
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model:int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.W = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor :
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)

        x = x / rms
        result = x * self.W

        return result.to(in_dtype)


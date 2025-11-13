import torch
from einops import einsum
from math import sqrt
class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features:int, device=None, dtype=None):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = sqrt(2 / (in_features + out_features))
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

import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange

def softmax(x: torch.Tensor, dim:int = -1) -> torch.Tensor :
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)

    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def cross_entropy(
        logits: Float[Tensor, "... vocab_size"], 
        targets: Int[Tensor, "..."]
        ) -> Float[Tensor, ""]:
    # 1. numerically stable: subtract max
    max_logits = torch.max(logits, dim = -1, keepdim=True).values

    # 2. logsumexp
    log_sum_exp_logits: Float[Tensor, "..."] = torch.log(torch.sum(torch.exp(logits - max_logits), dim=-1))

    # 3. gather correct logits
    targets = rearrange(targets, "... -> ... 1")
    correct_logits = rearrange(torch.gather(logits, dim=-1, index=targets) - max_logits, "... dim1 -> (... dim1)")

    return torch.mean(log_sum_exp_logits - correct_logits)

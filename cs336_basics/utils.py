import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange
from math import cos, pi
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

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    lr: float = 1
    if it < warmup_iters :
        lr = max_learning_rate * it / warmup_iters
    else :
        if it <= cosine_cycle_iters :
            lr = min_learning_rate + (max_learning_rate - min_learning_rate) / 2 * (1 + cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * pi))
        else :
            lr = min_learning_rate
    
    return lr

import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange
from math import cos, pi
from collections.abc import Iterable

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

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], 
    max_l2_norm: float, 
    epsilon: float = 1e-6
):
    # 步骤1：收集所有有效梯度（跳过无梯度或冻结的参数）
    grads = []
    for p in parameters:
        if p.grad is not None and p.requires_grad:
            # 展平梯度为1D向量，方便拼接计算全局范数
            grads.append(p.grad.flatten())
    
    if not grads:  # 没有需要裁剪的梯度，直接返回
        return
    
    # 步骤2：计算所有梯度的全局L2范数
    all_grads = torch.cat(grads)
    global_l2_norm = torch.linalg.norm(all_grads, ord=2)
    
    # 步骤3：若全局范数超过阈值，统一缩放所有梯度
    if global_l2_norm > max_l2_norm:
        scaling_factor = max_l2_norm / (global_l2_norm + epsilon)
        # 遍历参数，应用缩放因子
        for p in parameters:
            if p.grad is not None and p.requires_grad:
                p.grad.data *= scaling_factor  # 用 data 避免计算图问题

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)
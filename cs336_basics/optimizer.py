import torch
from typing import Optional
from collections.abc import Callable, Iterable
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, weight_decay=0.1, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            epsilon = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", 0)
                v = state.get("v", 0)
                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (torch.pow(grad,2))
                lr_t = lr * (math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
                p.data -= lr_t * m / (torch.sqrt(v) + epsilon)
                p.data -= lr * weight_decay * p.data

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        
        return loss


if __name__ == "__main__" :
    lr_list = [1e1, 1e2, 1e3]
    for lr in lr_list:
        print(f"Training with lr = {lr}")
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)
        for t in range(10):
            opt.zero_grad() # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            print(loss.cpu().item())
            loss.backward() # Run backward pass, which computes gradients.
            opt.step() # Run optimizer step.
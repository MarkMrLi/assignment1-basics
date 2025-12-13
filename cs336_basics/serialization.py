import torch
import os
import typing

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    checkpoint = {
        "iteration": iteration,
        "model_state_dict" : model.state_dict(),
        "optimizer_state_dict" : optimizer.state_dict()
    }
    
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int :
    checkpoint = torch.load(src)
    iteration = checkpoint["iteration"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return iteration

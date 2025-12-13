import torch
import numpy.typing as npt
import numpy as np

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    
    max_start_idx = dataset.shape[0] - context_length

    start_idx = np.random.choice(max_start_idx, size=batch_size, replace=False)

    train_data_np = np.stack([dataset[s:s + context_length] for s in start_idx])
    target_data_np = np.stack([dataset[s + 1:s + context_length + 1] for s in start_idx])

    train_data = torch.tensor(train_data_np, device=device)
    target_data = torch.tensor(target_data_np, device=device)

    return train_data, target_data
import numpy as np
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.module import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import cross_entropy, get_lr_cosine_schedule
from cs336_basics.data import get_batch
from cs336_basics.serialization import save_checkpoint, load_checkpoint
import argparse
from dotenv import load_dotenv
import os
from torch import Tensor
from pathlib import Path
import torch
load_dotenv()

def load_data(
    input_path: str,
    tokenizer: Tokenizer
) -> np.memmap :
    file_name = input_path.split('.')[0]
    extension = input_path.split('.')[1]
    
    if extension == "dat" :
        return np.memmap(input_path,dtype=np.int64,mode='r')
    
    # 1. 统计总 token 数
    total_tokens = count_token(input_path, tokenizer)
    shape = (total_tokens,)
    dtype = np.int64
    output_path = file_name + ".dat"
    memmap = np.memmap(output_path, dtype=dtype, shape=shape,mode='w+')

    idx = 0
    with open(input_path, 'r') as f:
        for line in f:
            tokens = tokenizer.encode(line)
            n_tokens = len(tokens)
            memmap[idx: idx + n_tokens] = tokens
            idx = idx + n_tokens
    memmap.flush()
    del memmap
    return np.memmap(output_path, dtype=dtype, mode="r")

def count_token(input_path, tokenizer:Tokenizer) -> int :
    total_tokens_len = 0
    with open(input_path, 'r') as f :
        for line in f:
            tokens = tokenizer.encode(line)
            total_tokens_len += len(tokens)
    
    return total_tokens_len


def main() :
    # 1. 加载 tokenizer
    tokenizer_file_path = os.getenv("TOKENIZER_PATH")
    tokenizer = Tokenizer.from_files(tokenizer_file_path)

    print("Finish preparing Tokenizer")
    # 2. 加载数据
    data_input_path = os.getenv("DATA_INPUT_PATH")
    data = load_data(data_input_path, tokenizer)
    print("Loaded Data")
    batch_size = int(os.getenv("BATCH_SIZE", "10000"))
    context_len = int(os.getenv("CONTEXT_LEN"))
    # 3. train_loop
    vocab_size = tokenizer.get_vocab_size()
    d_model = int(os.getenv("D_MODEL"))
    num_layers = int(os.getenv("NUM_LAYERS"))
    num_heads = int(os.getenv("NUM_HEADS"))
    d_ff = int(os.getenv("D_FF"))
    rope_theta = float(os.getenv("ROPE_THETA"))
    warmup_steps = int(os.getenv("WARMUP_STEPS"))
    max_steps = int(os.getenv("MAX_STEPS"))
    cosine_cycle_iters = int(os.getenv("COSINE_CYCLE_ITERS"))
    base_lr = float(os.getenv("BASE_LR"))
    min_lr = float(os.getenv("MIN_LR"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    optimizer = AdamW(params=model.parameters())

    serialization_path = Path(os.getenv("SERIALIZATION_PATH"))
    serialization_path.mkdir(parents=True, exist_ok=True)
    print("Start train loop")
    if device == "cuda":
        model.to("cuda")
    it = 0
    for it in range(max_steps):
        # 1. 计算 lr
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=base_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_steps,
            cosine_cycle_iters=cosine_cycle_iters,
        )

        # 2. 设置 optimizer 的 lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 3. 正常训练
        optimizer.zero_grad()
        x, y = get_batch(
            dataset=data, 
            batch_size=batch_size, 
            context_length=context_len,
            device = device
        )
        y_hat = model(x)
        loss = cross_entropy(logits=y_hat.view(-1,y_hat.size(-1)),targets=y.view(-1))
        loss.backward()
        optimizer.step()
        if it % 200 == 0 and it != 0:
            checkpoint_path = serialization_path / f"checkpoint_{it}.pt" 
            save_checkpoint(model=model, optimizer=optimizer, iteration=it,out=checkpoint_path)
            print(f"[checkpoint] saved to {checkpoint_path}")     

if __name__ == "__main__":
    main()
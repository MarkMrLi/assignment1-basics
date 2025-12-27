#!/usr/bin/env python3
"""
Profile the BPE training to identify bottlenecks (with memory analysis).
Usage: 
  - 内存分析: mprof run profile_bpe.py --memory
  - 其他分析: python profile_bpe.py [--line|--cprofile|--simple]
"""
import time
import tracemalloc
from pathlib import Path
from tests.adapters import run_train_bpe
from tests.common import FIXTURES_PATH
import pickle

# Import the functions we want to profile
from cs336_basics.tokenizer import (
    BPETokenizer,
    get_max_frequency_bytes_tuple,
    update_token_counts,
    pre_tokenization,
)

# 用 memory_profiler 的装饰器标记需要逐行分析内存的函数
from memory_profiler import profile

# 为核心函数添加内存分析装饰器
@profile
def profiled_run_train_bpe(*args, **kwargs):
    return run_train_bpe(*args, **kwargs)

@profile
def profiled_train_bpe_slow(self, *args, **kwargs):
    return BPETokenizer.train_bpe_slow(self, *args, **kwargs)

@profile
def profiled_get_max_frequency(*args, **kwargs):
    return get_max_frequency_bytes_tuple(*args, **kwargs)

@profile
def profiled_update_token_counts(*args, **kwargs):
    return update_token_counts(*args, **kwargs)

@profile
def profiled_pre_tokenization(*args, **kwargs):
    return pre_tokenization(*args, **kwargs)

# 替换原类方法为带装饰器的版本（确保内存分析生效）
BPETokenizer.train_bpe_slow = profiled_train_bpe_slow


def profile_with_memory_profiler():
    """使用 memory_profiler 分析内存占用（正确版本）- 新增前十长标记输出"""
    input_path = "/home/marklee/study/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    print(f"Profiling memory usage on {input_path}...")
    
    # 记录内存峰值（辅助统计）
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    
    start_time = time.time()
    # 调用带内存分析的函数
    vocab, merge_list = profiled_run_train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )
    elapsed = time.time() - start_time
    
    snapshot_after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    
    # 保存词汇表
    with open("bpe_data.pkl", "wb") as f:
        pickle.dump((vocab, merge_list), f)
    
    # 计算总内存占用（MB）
    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
    total_memory = sum(stat.size_diff for stat in top_stats) / (1024 * 1024)
    
    print(f"\nTotal time: {elapsed:.2f} seconds")
    print(f"Total memory used: {total_memory:.2f} MB")
    print("\n=== Top 10 Memory Growth Locations ===")
    for stat in top_stats[:10]:
        print(stat)
    
    # ---------------- 新增：输出词汇表前十长的标记 ----------------
    print("\n=== Top 10 Longest Tokens in Vocab (by Byte Length) ===")
    # 提取 (标记字节长度, 原始bytes标记) 列表
    token_with_length = [(len(token_bytes), token_bytes) for token_bytes in vocab.values()]
    # 按字节长度降序排序（越长越靠前）
    token_with_length.sort(reverse=True, key=lambda x: x[0])
    
    # 输出前10个最长标记
    for idx, (length, token_bytes) in enumerate(token_with_length[:10], 1):
        # 尝试解码为UTF-8字符串（失败则显示原始bytes）
        try:
            token_str = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            token_str = repr(token_bytes)  # 无法解码时显示原始格式
        print(f"{idx:2d}. 字节长度: {length:2d} | 标记: {token_str}")
    
    return True


def profile_with_line_profiler():
    """原有时间分析（保持不变）"""
    try:
        from line_profiler import LineProfiler
        
        lp = LineProfiler()
        lp.add_function(BPETokenizer.train_bpe_slow)
        lp.add_function(get_max_frequency_bytes_tuple)
        lp.add_function(update_token_counts)
        lp.add_function(pre_tokenization)
        
        lp_wrapper = lp(run_train_bpe)
        input_path = FIXTURES_PATH / "corpus.en"
        print(f"Profiling BPE training on {input_path}...")
        start = time.time()
        lp_wrapper(
            input_path=input_path,
            vocab_size=500,
            special_tokens=["<|endoftext|>"],
        )
        elapsed = time.time() - start
        print(f"\nTotal time: {elapsed:.2f} seconds")
        print(f"Target: < 1.5 seconds (reference: 0.38s)\n")
        lp.print_stats()
        
    except ImportError:
        print("line_profiler not installed. Install with: pip install line_profiler")
        return False
    
    return True


def profile_with_cprofile():
    """原有cProfile分析（保持不变）"""
    import cProfile
    import pstats
    from pstats import SortKey
    
    input_path = FIXTURES_PATH / "corpus.en"
    print(f"Profiling BPE training with cProfile on {input_path}...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    start = time.time()
    vocab, merge_list = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    elapsed = time.time() - start
    
    profiler.disable()
    print(f"\nTotal time: {elapsed:.2f} seconds")
    print(f"Target: < 1.5 seconds (reference: 0.38s)\n")
    
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats(SortKey.CUMULATIVE)
    print("\n=== Top 30 functions by cumulative time ===")
    stats.print_stats(30)
    
    stats.sort_stats(SortKey.TIME)
    print("\n=== Top 20 functions by total time ===")
    stats.print_stats(20)
    
    stats.dump_stats('bpe_profile.prof')
    print("\nProfile saved to bpe_profile.prof")
    print("View with: python -m pstats bpe_profile.prof")

    with open("bpe_data.pkl", "wb") as f:
        pickle.dump((vocab, merge_list), f)


def simple_timing_profile():
    """增强的简单分析（含内存）"""
    input_path = FIXTURES_PATH / "corpus.en"
    print(f"Running simple timing profile on {input_path}...")
    print("=" * 60)
    
    tracemalloc.start()
    snapshot_start = tracemalloc.take_snapshot()
    
    overall_start = time.time()
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    total_time = time.time() - overall_start
    
    snapshot_end = tracemalloc.take_snapshot()
    tracemalloc.stop()
    top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
    total_memory = sum(stat.size_diff for stat in top_stats) / (1024 * 1024)
    
    print(f"\nTotal time: {total_time:.3f} seconds")
    print(f"Total memory used: {total_memory:.2f} MB")
    print(f"Target: < 1.5 seconds (reference: 0.38s)")
    print(f"Status: {'✓ PASS' if total_time < 1.5 else '✗ FAIL'}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")


if __name__ == "__main__":
    import sys
    
    print("BPE Training Profiler (with Memory Analysis)")
    print("=" * 60)
    
    if "--memory" in sys.argv:
        print("\n1. Running memory_profiler (line-by-line memory)...")
        print("-" * 60)
        profile_with_memory_profiler()
        sys.exit(0)
    
    if "--line" in sys.argv or len(sys.argv) == 1:
        print("\n2. Running line_profiler (time)...")
        print("-" * 60)
        if profile_with_line_profiler():
            sys.exit(0)
    
    if "--cprofile" in sys.argv or len(sys.argv) == 1:
        print("\n3. Running cProfile (function time)...")
        print("-" * 60)
        profile_with_cprofile()
    
    if "--simple" in sys.argv or len(sys.argv) == 1:
        print("\n4. Running simple timing (with memory)...")
        print("-" * 60)
        simple_timing_profile()
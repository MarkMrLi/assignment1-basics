#!/usr/bin/env python3
"""
简单的实验结果可视化脚本
使用方法: python plot_results.py <实验目录路径>
例如: python plot_results.py logs/20240115_143022
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_experiment_results(experiment_path: str):
    """
    绘制实验结果
    
    Args:
        experiment_path: 实验目录路径
    """
    experiment_dir = Path(experiment_path)
    
    # 检查目录是否存在
    if not experiment_dir.exists():
        print(f"错误: 目录 {experiment_path} 不存在")
        return
    
    # 检查metrics文件是否存在
    metrics_file = experiment_dir / "metrics.csv"
    if not metrics_file.exists():
        print(f"错误: 找不到 metrics.csv 文件")
        return
    
    # 读取数据
    print(f"读取实验数据: {metrics_file}")
    df = pd.read_csv(metrics_file)
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 绘制损失曲线
    axes[0].plot(df['step'], df['loss'], linewidth=2)
    axes[0].set_xlabel('Training Step', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 绘制学习率曲线
    axes[1].plot(df['step'], df['lr'], linewidth=2, color='orange')
    axes[1].set_xlabel('Training Step', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = experiment_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {plot_path}")
    
    # 显示图表
    plt.show()
    
    # 打印统计信息
    print("\n=== 实验统计信息 ===")
    print(f"总训练步数: {len(df)}")
    print(f"初始损失: {df['loss'].iloc[0]:.4f}")
    print(f"最终损失: {df['loss'].iloc[-1]:.4f}")
    print(f"最低损失: {df['loss'].min():.4f} (步数: {df['loss'].idxmin()})")
    print(f"初始学习率: {df['lr'].iloc[0]:.6f}")
    print(f"最终学习率: {df['lr'].iloc[-1]:.6f}")

def list_all_experiments():
    """列出所有实验"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("没有找到logs目录")
        return
    
    experiments = sorted([d for d in logs_dir.iterdir() if d.is_dir()])
    if not experiments:
        print("logs目录中没有实验记录")
        return
    
    print("=== 可用的实验记录 ===")
    for exp in experiments:
        print(f"- {exp.name}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            list_all_experiments()
        else:
            plot_experiment_results(sys.argv[1])
    else:
        print("使用方法:")
        print("  查看所有实验: python plot_results.py --list")
        print("  可视化特定实验: python plot_results.py <实验目录名>")
        print("\n正在显示所有可用的实验...")
        print()
        list_all_experiments()
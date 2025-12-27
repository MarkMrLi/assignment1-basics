import csv
import logging
from pathlib import Path
from datetime import datetime
import json
import os

class SimpleLogger:
    def __init__(self, log_dir: str = "logs", experiment_name: str = None):
        """
        简单的实验日志记录器
        
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称，如果为None则自动生成时间戳名称
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        # 设置CSV文件记录训练指标
        self.metrics_file = self.experiment_dir / "metrics.csv"
        self.metrics_file_exists = self.metrics_file.exists()
        
        # 设置日志文件
        self.log_file = self.experiment_dir / "training.log"
        
        # 配置logging
        self._setup_logging()
        
        # 记录配置信息
        self.config_file = self.experiment_dir / "config.json"
        
    def _setup_logging(self):
        """配置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()  # 同时输出到控制台
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def log_config(self, config_dict: dict):
        """记录实验配置"""
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        self.logger.info(f"配置已保存到 {self.config_file}")
        
    def log_metrics(self, metrics_dict: dict, step: int):
        """
        记录训练指标到CSV文件
        
        Args:
            metrics_dict: 指标字典，如 {'loss': 3.2, 'lr': 0.001}
            step: 训练步数
        """
        # 准备写入的数据
        row_data = {'step': step, **metrics_dict}
        
        # 写入CSV
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            
            # 如果文件不存在，写入表头
            if not self.metrics_file_exists:
                writer.writeheader()
                self.metrics_file_exists = True
                
            writer.writerow(row_data)
            
    def log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(message)
        
    def log_checkpoint(self, step: int, checkpoint_path: str):
        """记录checkpoint信息"""
        self.logger.info(f"Checkpoint saved: step={step}, path={checkpoint_path}")
        
    def get_experiment_dir(self) -> Path:
        """获取实验目录路径"""
        return self.experiment_dir
        
    def print_summary(self):
        """打印实验摘要"""
        self.logger.info(f"实验目录: {self.experiment_dir}")
        self.logger.info(f"指标文件: {self.metrics_file}")
        self.logger.info(f"日志文件: {self.log_file}")
        self.logger.info(f"配置文件: {self.config_file}")
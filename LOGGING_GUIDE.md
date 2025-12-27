# 实验日志使用说明

## 快速开始

### 1. 配置环境变量
在您的`.env`文件中添加实验名称：
```
EXPERIMENT_NAME=transformer_experiment_1
```

### 2. 运行训练
训练时会自动创建日志文件，包含：
- `logs/<实验名称>/metrics.csv` - 训练指标数据
- `logs/<实验名称>/training.log` - 详细训练日志  
- `logs/<实验名称>/config.json` - 实验配置信息
- `logs/<实验名称>/training_curves.png` - 训练曲线图

### 3. 查看结果
```bash
# 列出所有实验
python plot_results.py --list

# 可视化特定实验的结果
python plot_results.py logs/<实验名称>
```

## 日志内容说明

### metrics.csv
每行包含一步的训练数据：
- `step` - 训练步数
- `loss` - 损失值
- `lr` - 学习率

### training.log  
包含训练过程中的所有日志信息：
- 时间戳
- 训练进度
- Checkpoint保存信息

### config.json
记录本次实验的所有配置参数，方便复现实验

## 数据分析

### 使用Python分析
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('logs/<实验名称>/metrics.csv')

# 绘制损失曲线
plt.plot(df['step'], df['loss'])
plt.xlabel('Step')
plt.ylabel('Loss')
plt.show()
```

### 使用Excel分析
直接用Excel打开`metrics.csv`文件，使用图表功能绘制曲线

## 多实验对比

多次实验会创建不同的目录，可以对比不同配置的效果：
```bash
# 运行不同配置的实验
python run_train.py  # 实验1
# 修改.env中的配置
python run_train.py  # 实验2

# 对比结果
python plot_results.py logs/<实验1名称>
python plot_results.py logs/<实验2名称>
```
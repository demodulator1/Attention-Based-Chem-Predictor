# 基于注意力机制的化学预测器
本代码库用于论文“基于注意力机制的神经网络在预测化学分子性质中的应用”。包括模型训练、评估和数据处理脚本。

[English](README.md)

## 项目结构

```
Attention-Based-Chem-Predictor/
│
├── edit-Attention-split.py       # 基于注意力机制的神经网络模型
├── edit-BP-split.py              # 反向传播神经网络模型
├── edit-RBF-split.py             # 径向基函数神经网络模型
├── draw.m                        # 用于绘制损失曲线的 MATLAB 脚本
├── predictions.xlsx              
├── predictions_optimized.xlsx    
├── LICENSE                       
├── README.md                     
├── README_zh.md                     
│
├── history/                      # 训练损失历史数据记录
│   ├── Attention_loss_history.txt
│   ├── BP_loss_history.txt
│   └── RBF_loss_history.txt
│
└── output/                       # 输出图表
    ├── Attention_loss_plot.png
    ├── BP_loss_plot.png
    └── RBF_loss_plot.png
```

## 描述

该项目实现并比较了三种不同的神经网络模型，用于预测化学分子性质：

1. 基于注意力机制的神经网络
2. 反向传播神经网络
3. 径向基函数神经网络

每个模型在单独的 Python 脚本中实现，并使用相同的数据集（`data.csv`）进行训练和评估。

## 使用方法

在运行任何脚本之前，请确保您已从 Releases 部分下载了 `data.csv` 文件，并将其放置在项目的根目录中。

### Python 脚本

要运行任何神经网络模型：

1. 确保已安装`pandas、numpy、tensorflow、scikit-learn`。
2. 运行所需的脚本：
```
python edit-Attention-split.py
python edit-BP-split.py
python edit-RBF-split.py
```

每个脚本将：
- 从 `data.csv` 加载并预处理数据
- 将数据划分为训练集和测试集
- 训练相应的神经网络模型
- 评估模型并将预测结果保存到 `predictions.xlsx`
- 将损失历史保存到 `history/` 目录中的文本文件

### MATLAB 脚本

要可视化训练和验证损失：

1. 打开 `draw.m` 并设置 `choose` 变量以选择所需的模型（1 表示注意力模型，2 表示 RBF，3 表示 BP）。
2. 运行该脚本。

该脚本将生成损失图并将其保存为 PNG 文件到 `output/` 目录中。

## 数据

`data.csv` 包含用于训练和测试模型的数据集。它包括 100 个特征列和 3 个目标列（y1、y2、y3）。

由于 GitHub 的文件大小限制，`data.csv` 文件未包含在主仓库中。您可以从本项目的 Releases 下载该文件。**请在运行脚本之前下载并将其放置在项目的根目录中。**

## 结果

训练过程生成的损失历史文件保存在 `history/` 目录中。这些文件包含每个 epoch 的训练和验证损失。

`output/` 目录包含每个模型的损失曲线的 PNG 图像，这些图像通过 MATLAB 脚本生成。

## 许可

本项目根据 Apache 许可证 2.0 版进行许可。详细信息请参见 `LICENSE` 文件。


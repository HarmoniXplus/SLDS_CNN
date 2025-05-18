# CNN手写数字识别实验项目

这个项目实现了基于PyTorch的CNN（卷积神经网络）模型，用于MNIST手写数字识别任务。项目的主要目标是分析CNN模型超参数、网络深度和网络结构对性能的影响。

## 项目结构

```
CNN/
├── config.py               # 配置参数
├── main.py                 # 主程序入口
├── train.py                # 训练脚本
├── test.py                 # 测试脚本
├── models/                 # 模型定义
│   └── cnn_model.py        # CNN模型类
├── utils/                  # 工具函数
│   ├── data_loader.py      # 数据加载
│   ├── training.py         # 训练工具
│   ├── visualization.py    # 可视化工具
│   └── model_visualization.py # 模型结构可视化
├── PlotNeuralNet/          # 神经网络结构可视化工具
├── checkpoints/            # 模型保存目录
├── figures/                # 图表保存目录
└── data/                   # 数据集保存目录
```

## 功能特点

1. **多种网络架构**：
   - 简单CNN模型，支持配置深度和宽度
   - 多层感知机(MLP)模型，用于对比传统全连接网络和CNN
   - 基于残差连接的ResNet模型

2. **超参数分析**：
   - 网络深度比较
   - 卷积核数量/隐藏层大小比较
   - 残差连接结构比较

3. **完整的训练/验证/测试流程**：
   - 自动下载并处理MNIST数据集
   - 训练过程中保存最佳模型
   - 绘制损失和准确率曲线

4. **结果可视化**：
   - 训练/验证/测试损失曲线
   - 训练/验证/测试准确率曲线
   - 混淆矩阵分析
   - 卷积层特征图可视化
   - 错误分类样本分析
   - 模型结构可视化

## 环境需求

- Python 3.6+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- scikit-learn
- seaborn
- tqdm
- LaTeX (用于模型结构可视化，可选)
- ImageMagick (用于PDF转PNG，可选)

可通过以下命令安装所需依赖：

```bash
pip install torch torchvision numpy matplotlib scikit-learn seaborn tqdm
```

对于模型结构可视化功能，需要安装PlotNeuralNet工具：

```bash
git clone https://github.com/HarisIqbal88/PlotNeuralNet.git
```

## 使用方法

### 1. 训练模型

训练默认的简单CNN模型：

```bash
python main.py train
```

训练指定类型和配置的模型：

```bash
python main.py train --model_type CNN --config medium --epochs 20
```

训练MLP模型：

```bash
python main.py train --model_type mlp --config medium --epochs 20
```

比较多个模型配置：

```bash
python main.py train --compare
```

### 2. 测试模型

测试默认模型：

```bash
python main.py test
```

测试指定模型：

```bash
python main.py test --model_type resnet --config complex --model_path ./checkpoints/my_model.pth
```

### 3. 可视化卷积特征

```bash
python main.py visualize --model_type CNN
```

### 4. 可视化模型结构

使用PlotNeuralNet工具生成模型结构图（需要LaTeX环境）：

```bash
python main.py structure --model_type CNN --config medium
```

生成MLP模型结构图：

```bash
python main.py structure --model_type mlp --config complex
```

### 5. 一键执行完整流程

训练、测试和可视化全部执行：

```bash
python main.py all --model_type CNN --config medium
```

## 模型配置说明

### 简单CNN (SimpleCNN)

项目提供三种预设配置：

- **simple**: 2层卷积，初始32个滤波器
- **medium**: 3层卷积，初始64个滤波器
- **complex**: 4层卷积，初始128个滤波器

### 多层感知机 (MLP)

项目提供三种预设配置：

- **simple**: 单隐藏层 [128]，dropout率0.3
- **medium**: 两个隐藏层 [256, 128]，dropout率0.5
- **complex**: 三个隐藏层 [512, 256, 128]，dropout率0.5

### 残差网络 (ResNet)

项目提供三种预设配置：

- **simple**: 每层1个残差块 [1,1,1]
- **medium**: 每层2个残差块 [2,2,2]
- **complex**: 更深的残差结构 [3,4,6]

## 结果分析

运行完整的实验后，可以在`figures/`目录找到以下结果：

1. 每个模型的学习曲线（损失和准确率）
2. 不同模型对比图
3. 混淆矩阵
4. 卷积层特征图（仅CNN和ResNet模型）
5. 错误分类的样本
6. 模型结构图（需要LaTeX环境）

这些可视化结果将帮助分析各种网络结构的特征提取能力和性能差异，并对比传统MLP和CNN在图像识别任务上的效果差异。

## 模型结构可视化

项目使用PlotNeuralNet工具生成清晰美观的神经网络结构图，支持以下功能：

1. 自动根据配置生成动态结构图
2. 支持三种模型类型（SimpleCNN、MLP、ResNet）
3. 生成PDF或PNG格式的结构图
4. 可视化卷积层、池化层、全连接层、批归一化层等

注意：模型结构可视化功能需要LaTeX环境。如果没有安装LaTeX，将只生成.tex文件，需要手动编译生成PDF。 
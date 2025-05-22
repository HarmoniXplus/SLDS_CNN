"""
可视化工具模块
包含损失曲线、准确率曲线、混淆矩阵和卷积特征可视化函数
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from config import FIGURE_PATH, CLASS_NAMES

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建图像保存目录
os.makedirs(FIGURE_PATH, exist_ok=True)


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, title=None, save_path=None):
    """
    绘制学习曲线（损失和准确率）
    
    参数:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_accs: 训练准确率列表
        val_accs: 验证准确率列表
        title: 图表标题
        save_path: 保存路径，如果不提供则显示图表
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('损失曲线')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.title('准确率曲线')
    plt.legend()
    
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(cm, class_names):
    """
    绘制混淆矩阵
    
    参数:
        cm: 混淆矩阵
        class_names: 类别名称列表
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()


def plot_model_comparison(results, metric='accuracy', title=None, save_path=None):
    """
    绘制模型比较图
    
    参数:
        results: 包含模型性能的字典
        metric: 要比较的指标 ('accuracy' 或 'loss')
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    # 提取数据
    model_names = list(results.keys())
    if metric == 'accuracy':
        values = [results[name]['val_acc'] for name in model_names]
        ylabel = '验证准确率'
    else:
        values = [results[name]['val_loss'] for name in model_names]
        ylabel = '验证损失'
    
    # 绘制柱状图
    plt.bar(model_names, values)
    plt.xlabel('模型')
    plt.ylabel(ylabel)
    
    if title:
        plt.title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_feature_maps(model, image, device, layer_idx=0, title=None, save_path=None):
    """
    可视化卷积层特征图（紧凑排列所有特征图，不显示标签，不显示原图）
    """
    activation = {}
    def hook_fn(module, input, output):
        activation['feature'] = output.detach().cpu()
    conv_layers = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append(module)
    if layer_idx >= len(conv_layers):
        print(f"Warning: layer_idx {layer_idx} out of range. Using last conv layer.")
        layer_idx = len(conv_layers) - 1
    handle = conv_layers[layer_idx].register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        _ = model(image.unsqueeze(0).to(device))
    feature_maps = activation['feature'].squeeze(0)
    handle.remove()
    num_maps = feature_maps.size(0)
    grid_cols = int(np.ceil(np.sqrt(num_maps)))
    grid_rows = int(np.ceil(num_maps / grid_cols))
    plt.figure(figsize=(grid_cols * 2, grid_rows * 2))
    for i in range(num_maps):
        plt.subplot(grid_rows, grid_cols, i + 1)
        plt.imshow(feature_maps[i].cpu(), cmap='viridis')
        plt.axis('off')
    if title:
        plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Feature maps visualization saved to {save_path}")
    else:
        plt.show()


def visualize_all_conv_layers(model, image, device, save_dir=None):
    """
    可视化所有卷积层的特征图（每层所有特征图紧凑排列，无标签）
    """
    conv_layers = []
    for i, module in enumerate(model.modules()):
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append((i, module))
    print(f"Found {len(conv_layers)} convolutional layers")
    for i, (layer_idx, _) in enumerate(conv_layers):
        if save_dir:
            save_path = os.path.join(save_dir, f'conv_layer_{i}.png')
        else:
            save_path = None
        visualize_feature_maps(
            model, image, device, i,
            title=None,
            save_path=save_path
        )


def visualize_filters(model, layer_idx=0, title=None, save_path=None):
    """
    可视化卷积层滤波器
    
    参数:
        model: CNN模型
        layer_idx: 要可视化的卷积层索引
        title: 图表标题
        save_path: 保存路径，如果不提供则显示图表
    """
    # 获取所有卷积层
    conv_layers = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append(module)
    
    # 确保layer_idx有效
    if layer_idx >= len(conv_layers):
        print(f"Warning: layer_idx {layer_idx} out of range. Using last conv layer.")
        layer_idx = len(conv_layers) - 1
    
    # 获取滤波器权重
    filters = conv_layers[layer_idx].weight.detach().cpu().numpy()
    
    # 确定绘图布局
    num_filters = filters.shape[0]
    num_channels = filters.shape[1]
    
    # 每个滤波器显示所有通道
    if num_channels == 1:  # 单通道滤波器
        grid_size = int(np.ceil(np.sqrt(min(num_filters, 16))))
        plt.figure(figsize=(15, 15))
        for i in range(min(num_filters, grid_size**2)):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(filters[i, 0], cmap='viridis')
            plt.title(f"Filter {i}")
            plt.axis('off')
    else:  # 多通道滤波器，为每个滤波器显示第一个通道
        grid_size = int(np.ceil(np.sqrt(min(num_filters, 16))))
        plt.figure(figsize=(15, 15))
        for i in range(min(num_filters, grid_size**2)):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(filters[i, 0], cmap='viridis')
            plt.title(f"Filter {i}, Channel 0")
            plt.axis('off')
    
    # 设置总标题
    if title:
        plt.suptitle(title, fontsize=16)
    else:
        plt.suptitle(f'Filters from Conv Layer {layer_idx}', fontsize=16)
    
    plt.tight_layout()
    
    # 保存或显示图表
    if save_path:
        plt.savefig(save_path)
        print(f"Filter visualization saved to {save_path}")
    else:
        plt.show()


def visualize_misclassified(model, dataloader, device, num_images=10, save_path=None):
    """
    可视化错误分类的样本
    
    参数:
        model: 模型实例
        dataloader: 数据加载器
        device: 设备
        num_images: 要可视化的图像数量
        save_path: 保存路径，如果不提供则显示图表
    """
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            # 找出错误分类的样本
            mask = predicted != labels
            if mask.any():
                misclassified_images.extend(images[mask].cpu())
                misclassified_labels.extend(labels[mask].cpu())
                misclassified_preds.extend(predicted[mask].cpu())
            
            if len(misclassified_images) >= num_images:
                break
    
    # 限制图像数量
    misclassified_images = misclassified_images[:num_images]
    misclassified_labels = misclassified_labels[:num_images]
    misclassified_preds = misclassified_preds[:num_images]
    
    # 创建图表
    plt.figure(figsize=(15, 5))
    for i, (image, true_label, pred_label) in enumerate(zip(misclassified_images, misclassified_labels, misclassified_preds)):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image.squeeze().numpy(), cmap='gray')
        plt.title(f'真实: {true_label}\n预测: {pred_label}')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 
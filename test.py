"""
测试脚本
测试已训练的模型并生成混淆矩阵
"""
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from config import MODEL_PATH, FIGURE_PATH, CLASS_NAMES, MODEL_CONFIGS, MLP_CONFIGS
from models.cnn_model import get_cnn_model
from utils.data_loader import get_dataloaders
from utils.training import Trainer
from utils.visualization import (
    plot_confusion_matrix, 
    visualize_feature_maps, 
    visualize_all_conv_layers,
    visualize_misclassified,
    plot_learning_curves
)


def test_model(model_path, model_type, config, dataset='mnist', class_names=None):
    """
    测试模型
    
    参数:
        model_path: 模型文件路径
        model_type: 模型类型
        config: 模型配置
        dataset: 数据集名称
        class_names: 类别名称列表
    
    返回:
        test_loss: 测试损失
        test_acc: 测试准确率
        all_preds: 所有预测结果
        all_labels: 所有真实标签
    """
    # 获取数据加载器
    _, _, test_loader = get_dataloaders(dataset)
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    if model_type == 'CNN':
        model = get_cnn_model(
            model_type=model_type,
            num_filters=config['num_filters'],
            num_layers=config['num_layers']
        )
    elif model_type == 'resnet':
        model = get_cnn_model(
            model_type=model_type,
            num_blocks=config['num_blocks']
        )
    elif model_type == 'mlp':
        model = get_cnn_model(
            model_type=model_type,
            hidden_sizes=config['hidden_sizes'],
            dropout_rate=config['dropout_rate']
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 创建训练器
    trainer = Trainer(model=model, device=device)
    
    # 加载模型
    trainer.load_model(model_path)
    
    # 测试模型
    test_loss, test_acc, all_preds, all_labels = trainer.test(test_loader)
    
    return test_loss, test_acc, all_preds, all_labels


def visualize_model_features(model_path, model_type, config, dataset='mnist'):
    """
    可视化模型特征
    
    参数:
        model_path: 模型文件路径
        model_type: 模型类型
        config: 模型配置
        dataset: 数据集名称
    """
    # 获取数据加载器
    _, _, test_loader = get_dataloaders(dataset)
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    if model_type == 'CNN':
        model = get_cnn_model(
            model_type=model_type,
            num_filters=config['num_filters'],
            num_layers=config['num_layers']
        )
    elif model_type == 'resnet':
        model = get_cnn_model(
            model_type=model_type,
            num_blocks=config['num_blocks']
        )
    elif model_type == 'mlp':
        model = get_cnn_model(
            model_type=model_type,
            hidden_sizes=config['hidden_sizes'],
            dropout_rate=config['dropout_rate']
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 创建训练器
    trainer = Trainer(model=model, device=device)
    
    # 加载模型
    trainer.load_model(model_path)
    
    # 获取一批测试数据
    images, labels = next(iter(test_loader))
    images = images.to(device)
    
    # 获取模型特征
    model.eval()
    with torch.no_grad():
        features = model.get_features(images)
    
    # 可视化特征
    if model_type == 'CNN':
        visualize_cnn_features(features, labels)
    elif model_type == 'resnet':
        visualize_resnet_features(features, labels)
    elif model_type == 'mlp':
        visualize_mlp_features(features, labels)


def visualize_cnn_features(features, labels):
    """可视化CNN模型特征"""
    # 获取特征图
    feature_maps = features.cpu().numpy()
    
    # 选择一些样本进行可视化
    num_samples = min(5, len(feature_maps))
    num_channels = min(16, feature_maps.shape[1])
    
    # 创建图表
    fig, axes = plt.subplots(num_samples, num_channels, figsize=(20, 4))
    
    for i in range(num_samples):
        for j in range(num_channels):
            if num_samples == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            
            # 显示特征图
            ax.imshow(feature_maps[i, j], cmap='viridis')
            ax.axis('off')
            
            # 添加标签
            if i == 0:
                ax.set_title(f'Channel {j+1}')
    
    plt.tight_layout()
    plt.show()


def visualize_resnet_features(features, labels):
    """可视化ResNet模型特征"""
    # 获取特征向量
    feature_vectors = features.cpu().numpy()
    
    # 使用t-SNE降维
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(feature_vectors)
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                         c=labels.cpu().numpy(), cmap='tab10')
    plt.colorbar(scatter)
    plt.title('ResNet特征可视化 (t-SNE)')
    plt.show()


def visualize_mlp_features(features, labels):
    """可视化MLP模型特征"""
    # 获取特征向量
    feature_vectors = features.cpu().numpy()
    
    # 使用PCA降维
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(feature_vectors)
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                         c=labels.cpu().numpy(), cmap='tab10')
    plt.colorbar(scatter)
    plt.title('MLP特征可视化 (PCA)')
    plt.show()


if __name__ == "__main__":
    # 测试模型
    test_loss, test_acc, all_preds, all_labels = test_model(".\checkpoints\simple_simple_best.pth", "CNN", MODEL_CONFIGS['simple'])
    
    # 可视化模型特征（如果是CNN模型）
    model_type = 'CNN'  # 默认模型类型，实际应从命令行参数获取
    if model_type != 'mlp':
        visualize_model_features(".\checkpoints\simple_simple_best.pth", "CNN", MODEL_CONFIGS['simple'])

    # 绘制学习曲线
    plot_learning_curves(MODEL_PATH, model_type, MODEL_CONFIGS[model_type])
    
    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, CLASS_NAMES) 
"""
训练脚本
训练不同网络深度和结构的CNN模型
"""
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import MODEL_CONFIGS, MLP_CONFIGS, EPOCHS, LEARNING_RATE, FIGURE_PATH, SAVE_PATH
from models.cnn_model import get_cnn_model
from utils.data_loader import get_dataloaders
from utils.training import Trainer, train_multiple_models
from utils.visualization import plot_learning_curves, plot_model_comparison

# 确保图像保存路径存在
os.makedirs(FIGURE_PATH, exist_ok=True)


def train_single_model(model_type='CNN', config_name='simple', dataset='mnist', epochs=EPOCHS):
    """
    训练单个模型
    
    参数:
        model_type: 模型类型 ('CNN', 'resnet', 或 'mlp')
        config_name: 模型配置名称 ('simple', 'medium', 或 'complex')
        dataset: 数据集名称 ('mnist' 或 'ddr')
        epochs: 训练轮次
    
    返回:
        trainer: 训练器对象
    """
    print(f"\n=== 训练 {model_type} ({config_name}) 模型 ===")
    
    # 获取数据加载器
    train_loader, val_loader, _ = get_dataloaders(dataset)
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    if model_type == 'CNN':
        config = MODEL_CONFIGS[config_name]
        model = get_cnn_model(
            model_type=model_type,
            num_filters=config['num_filters'],
            num_layers=config['num_layers']
        )
    elif model_type == 'resnet':
        if config_name == 'simple':
            config = {'num_blocks': [1, 1, 1]}
        elif config_name == 'medium':
            config = {'num_blocks': [2, 2, 2]}
        else:  # complex
            config = {'num_blocks': [3, 4, 6]}
        model = get_cnn_model(
            model_type=model_type,
            num_blocks=config['num_blocks']
        )
    elif model_type == 'mlp':
        config = MLP_CONFIGS[config_name]
        model = get_cnn_model(
            model_type=model_type,
            hidden_sizes=config['hidden_sizes'],
            dropout_rate=config['dropout_rate']
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    print(f"创建模型: {model_type} ({config_name})")
    print(f"参数总数: {sum(p.numel() for p in model.parameters())}")
    
    # 创建训练器，指定模型保存路径
    save_path = os.path.join(SAVE_PATH, f"{model_type}_{config_name}")
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=LEARNING_RATE,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # 训练模型
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_path=save_path
    )
    
    # 绘制学习曲线
    plot_learning_curves(
        trainer.train_losses, trainer.val_losses,
        trainer.train_accs, trainer.val_accs,
        title=f"{model_type.capitalize()} ({config_name}) 模型学习曲线",
        save_path=os.path.join(FIGURE_PATH, f"{model_type}_{config_name}_learning_curves.png")
    )
    
    return trainer


def train_compare_models(dataset='mnist', epochs=EPOCHS):
    """
    训练并比较多个模型配置
    
    参数:
        dataset: 数据集名称 ('mnist' 或 'ddr')
        epochs: 训练轮次
    
    返回:
        results: 训练结果字典
    """
    print("\n=== 训练并比较多个模型配置 ===")
    
    # 要比较的模型配置
    model_configs = [
        ('CNN', 'simple'),
        ('CNN', 'medium'),
        ('CNN', 'complex'),
        ('resnet', 'simple'),
        ('resnet', 'medium'),
        ('resnet', 'complex'),
        ('mlp', 'simple'),
        ('mlp', 'medium'),
        ('mlp', 'complex')
    ]
    
    # 存储训练结果
    results = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': []  # 新增测试集准确率
    }
    
    # 获取测试数据加载器
    _, _, test_loader = get_dataloaders(dataset)
    
    # 训练每个模型配置
    for model_type, config_name in model_configs:
        print(f"\n训练 {model_type} ({config_name}) 模型...")
        trainer = train_single_model(
            model_type=model_type,
            config_name=config_name,
            dataset=dataset,
            epochs=epochs
        )
        
        # 记录训练结果
        results['train_loss'].append(trainer.train_losses)
        results['val_loss'].append(trainer.val_losses)
        results['train_acc'].append(trainer.train_accs)
        results['val_acc'].append(trainer.val_accs)
        
        # 加载最后一轮模型并测试
        model_path = os.path.join(SAVE_PATH, f"{model_type}_{config_name}_last.pth")
        trainer.load_model(model_path)
        test_loss, test_acc, _, _, _, _ = trainer.test(test_loader)
        results['test_acc'].append(test_acc)
        print(f"{model_type} ({config_name}) 测试集准确率: {test_acc:.4f}")
    
    # 绘制比较图表
    plot_comparison(results, model_configs, dataset)
    
    return results


def plot_comparison(results, model_configs, dataset):
    """
    绘制模型比较图表
    
    参数:
        results: 训练结果字典
        model_configs: 模型配置列表
        dataset: 数据集名称
    """
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制损失曲线
    for i, (model_type, config_name) in enumerate(model_configs):
        label = f"{model_type} ({config_name})"
        axes[0, 0].plot(results['train_loss'][i], label=label)
        axes[0, 1].plot(results['val_loss'][i], label=label)
        axes[1, 0].plot(results['train_acc'][i], label=label)
        axes[1, 1].plot(results['val_acc'][i], label=label)
    
    # 设置图表标题和标签
    axes[0, 0].set_title('训练损失')
    axes[0, 1].set_title('验证损失')
    axes[1, 0].set_title('训练准确率')
    axes[1, 1].set_title('验证准确率')
    
    for ax in axes.flat:
        ax.set_xlabel('轮次')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, f"model_learning_curves_{dataset}.png"))
    plt.close()
    
    # 提取每个模型最后一轮的训练准确率和验证准确率
    final_train_accs = [train_acc_list[-1] for train_acc_list in results['train_acc']]
    final_val_accs = [val_acc_list[-1] for val_acc_list in results['val_acc']]
    test_accs = results['test_acc']
    
    # 创建准确率比较直方图
    model_labels = [f"{model_type} ({config_name})" for model_type, config_name in model_configs]
    
    # 设置图表大小
    plt.figure(figsize=(15, 8))
    
    # 设置柱状图的位置
    x = np.arange(len(model_labels))
    width = 0.25
    
    # 按照用户要求，将每个模型的三个准确率放在一起
    # 创建分组柱状图
    n_groups = len(model_labels)
    index = np.arange(n_groups)
    bar_width = 0.25
    
    # 绘制三种准确率的柱状图
    plt.bar(index, final_train_accs, bar_width, label='训练准确率')
    plt.bar(index + bar_width, final_val_accs, bar_width, label='验证准确率')
    plt.bar(index + 2 * bar_width, test_accs, bar_width, label='测试准确率')
    
    # 添加标题和标签
    plt.title('模型准确率比较')
    plt.xlabel('模型配置')
    plt.ylabel('准确率')
    plt.ylim(0.9, 1.0)
    plt.xticks(index + bar_width, model_labels, rotation=45, ha='right')
    plt.legend()
    
    # 在柱状图上方添加数值标签
    for i, v in enumerate(final_train_accs):
        plt.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(final_val_accs):
        plt.text(i + bar_width, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(test_accs):
        plt.text(i + 2 * bar_width, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, f"model_accuracy_comparison_{dataset}.png"))
    print(f"模型准确率比较图已保存到 {FIGURE_PATH}/model_accuracy_comparison_{dataset}.png")
    plt.close()
    
    # 创建热力图
    model_types = ['CNN', 'resnet', 'mlp']
    complexities = ['simple', 'medium', 'complex']
    
    # 创建3x3的准确率矩阵
    acc_matrix = np.zeros((3, 3))
    for i, (model_type, config_name) in enumerate(model_configs):
        row = model_types.index(model_type)
        col = complexities.index(config_name)
        acc_matrix[row, col] = test_accs[i]
    
    # 创建热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(acc_matrix, 
                annot=True, 
                fmt='.4f',
                cmap='YlOrRd',
                xticklabels=complexities,
                yticklabels=model_types)
    
    plt.title('不同模型在测试集上的准确率')
    plt.xlabel('模型复杂度')
    plt.ylabel('模型类型')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_PATH, f"model_accuracy_heatmap_{dataset}.png"))
    print(f"模型准确率热力图已保存到 {FIGURE_PATH}/model_accuracy_heatmap_{dataset}.png")
    plt.close()


if __name__ == "__main__":
    # 如果只想训练单个模型，取消下面的注释并选择合适的配置
    # trainer = train_single_model(model_type='CNN', config_name='medium', epochs=EPOCHS)
    
    # 训练并比较多个模型
    results = train_compare_models(dataset='mnist', epochs=EPOCHS) 
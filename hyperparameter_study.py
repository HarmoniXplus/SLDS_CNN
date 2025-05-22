"""
超参数研究
比较CNN中等复杂度模型上的不同超参数设置
1. 学习率：比较固定学习率，使用不同学习率优化器
2. 比较使用不同优化器
3. 比较不同训练轮次
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from config import SAVE_PATH, FIGURE_PATH, USE_AMP
from models.cnn_model import get_cnn_model
from utils.data_loader import get_dataloaders
from utils.training import Trainer

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建实验结果保存路径
EXPERIMENT_PATH = os.path.join(FIGURE_PATH, 'hyperparameter_study')
os.makedirs(EXPERIMENT_PATH, exist_ok=True)

# 实验配置
LR_CONFIGS = {
    'fixed': [0.001, 0.01, 0.1],
    'scheduler': {
        'StepLR': {'step_size': 5, 'gamma': 0.1},
        'ReduceLROnPlateau': {'patience': 3, 'factor': 0.1},
        'CosineAnnealingLR': {'T_max': 10}
    }
}

OPTIMIZER_CONFIGS = {
    'SGD': {'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4},
    'Adam': {'lr': 0.001, 'weight_decay': 1e-4},
    'RMSprop': {'lr': 0.001, 'weight_decay': 1e-4}
}

EPOCH_CONFIGS = [5, 10, 15, 20]  # 为了快速实验，使用较小的轮次

class HyperparameterTrainer(Trainer):
    """
    超参数实验训练器
    扩展Trainer类，添加学习率和优化器选择功能
    """
    def __init__(self, model, device=None, optimizer_name='SGD', optimizer_params=None, 
                 scheduler_name=None, scheduler_params=None):
        """
        初始化超参数实验训练器
        
        参数:
            model: 要训练的模型
            device: 训练设备
            optimizer_name: 优化器名称 ('SGD', 'Adam', 'RMSprop')
            optimizer_params: 优化器参数
            scheduler_name: 学习率调度器名称 (None, 'StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR')
            scheduler_params: 学习率调度器参数
        """
        # 初始化基本属性
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 创建损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 创建梯度缩放器（用于混合精度训练）
        self.scaler = GradScaler() if USE_AMP else None
        
        # 训练记录
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.best_val_acc = 0
        self.lr_history = []  # 记录学习率历史
        
        # 确保模型保存路径存在
        os.makedirs(SAVE_PATH, exist_ok=True)
        
        # 创建优化器
        optimizer_params = optimizer_params or {}
        if optimizer_name == 'SGD':
            self.optimizer = optim.SGD(model.parameters(), **optimizer_params)
        elif optimizer_name == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), **optimizer_params)
        elif optimizer_name == 'RMSprop':
            self.optimizer = optim.RMSprop(model.parameters(), **optimizer_params)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        # 创建学习率调度器
        self.scheduler_name = scheduler_name
        scheduler_params = scheduler_params or {}
        
        if scheduler_name == 'StepLR':
            self.scheduler = StepLR(self.optimizer, **scheduler_params)
        elif scheduler_name == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', **scheduler_params)
        elif scheduler_name == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(self.optimizer, **scheduler_params)
        elif scheduler_name is not None:
            raise ValueError(f"不支持的学习率调度器: {scheduler_name}")
        else:
            self.scheduler = None
    
    def train(self, train_loader, val_loader, epochs, save_path):
        """
        训练模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮次
            save_path: 模型保存路径
        
        返回:
            history: 训练历史
        """
        print(f"\n开始训练...")
        print(f"模型将保存到: {save_path}")
        
        for epoch in range(epochs):
            # 记录当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)
            
            # 训练一个轮次
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self._validate(val_loader)
            
            # 更新学习率
            if self.scheduler_name == 'ReduceLROnPlateau':
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()
            
            # 记录训练历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # 打印训练信息
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(f"{save_path}_best.pth")
            
            # 保存最后一轮模型
            if epoch == epochs - 1:
                self.save_model(f"{save_path}_last.pth")
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_acc': self.train_accs,
            'val_acc': self.val_accs,
            'lr_history': self.lr_history
        }


def run_lr_experiment(dataset='mnist', epochs=10):
    """
    运行学习率实验
    
    参数:
        dataset: 数据集名称
        epochs: 训练轮次
        
    返回:
        results: 实验结果字典
    """
    print("\n=== 运行学习率实验 ===")
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(dataset)
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 存储结果
    results = {}
    
    # 1. 固定学习率实验
    for lr in LR_CONFIGS['fixed']:
        print(f"\n--- 固定学习率: {lr} ---")
        
        # 创建模型
        model = get_cnn_model(model_type='CNN', num_filters=32, num_layers=3)
        
        # 创建训练器
        trainer = HyperparameterTrainer(
            model=model,
            device=device,
            optimizer_name='SGD',
            optimizer_params={'lr': lr, 'momentum': 0.9, 'weight_decay': 1e-4}
        )
        
        # 训练模型
        save_path = os.path.join(SAVE_PATH, f"lr_fixed_{lr}")
        history = trainer.train(train_loader, val_loader, epochs, save_path)
        
        # 测试模型
        test_loss, test_acc, _, _, _, _ = trainer.test(test_loader)
        print(f"测试集准确率: {test_acc:.4f}")
        
        # 保存结果
        results[f'fixed_lr_{lr}'] = {
            'history': history,
            'test_acc': test_acc,
            'params': {'lr': lr}
        }
    
    # 2. 学习率调度器实验
    for scheduler_name, params in LR_CONFIGS['scheduler'].items():
        print(f"\n--- 学习率调度器: {scheduler_name} ---")
        
        # 创建模型
        model = get_cnn_model(model_type='CNN', num_filters=32, num_layers=3)
        
        # 创建训练器
        trainer = HyperparameterTrainer(
            model=model,
            device=device,
            optimizer_name='SGD',
            optimizer_params={'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4},
            scheduler_name=scheduler_name,
            scheduler_params=params
        )
        
        # 训练模型
        save_path = os.path.join(SAVE_PATH, f"lr_scheduler_{scheduler_name}")
        history = trainer.train(train_loader, val_loader, epochs, save_path)
        
        # 测试模型
        test_loss, test_acc, _, _, _, _ = trainer.test(test_loader)
        print(f"测试集准确率: {test_acc:.4f}")
        
        # 保存结果
        results[f'scheduler_{scheduler_name}'] = {
            'history': history,
            'test_acc': test_acc,
            'params': params
        }
    
    # 绘制结果
    plot_lr_comparison(results, epochs)
    
    return results


def run_optimizer_experiment(dataset='mnist', epochs=10):
    """
    运行优化器实验
    
    参数:
        dataset: 数据集名称
        epochs: 训练轮次
        
    返回:
        results: 实验结果字典
    """
    print("\n=== 运行优化器实验 ===")
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(dataset)
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 存储结果
    results = {}
    
    # 优化器实验
    for opt_name, params in OPTIMIZER_CONFIGS.items():
        print(f"\n--- 优化器: {opt_name} ---")
        
        # 创建模型
        model = get_cnn_model(model_type='CNN', num_filters=32, num_layers=3)
        
        # 创建训练器
        trainer = HyperparameterTrainer(
            model=model,
            device=device,
            optimizer_name=opt_name,
            optimizer_params=params
        )
        
        # 训练模型
        save_path = os.path.join(SAVE_PATH, f"optimizer_{opt_name}")
        history = trainer.train(train_loader, val_loader, epochs, save_path)
        
        # 测试模型
        test_loss, test_acc, _, _, _, _ = trainer.test(test_loader)
        print(f"测试集准确率: {test_acc:.4f}")
        
        # 保存结果
        results[opt_name] = {
            'history': history,
            'test_acc': test_acc,
            'params': params
        }
    
    # 绘制结果
    plot_optimizer_comparison(results, epochs)
    
    return results


def run_epoch_experiment(dataset='mnist'):
    """
    运行训练轮次实验
    
    参数:
        dataset: 数据集名称
        
    返回:
        results: 实验结果字典
    """
    print("\n=== 运行训练轮次实验 ===")
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(dataset)
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 存储结果
    results = {}
    
    # 训练轮次实验
    for epochs in EPOCH_CONFIGS:
        print(f"\n--- 训练轮次: {epochs} ---")
        
        # 创建模型
        model = get_cnn_model(model_type='CNN', num_filters=32, num_layers=3)
        
        # 创建训练器
        trainer = HyperparameterTrainer(
            model=model,
            device=device,
            optimizer_name='SGD',
            optimizer_params={'lr': 0.001, 'momentum': 0.9, 'weight_decay': 1e-4}
        )
        
        # 训练模型
        save_path = os.path.join(SAVE_PATH, f"epochs_{epochs}")
        history = trainer.train(train_loader, val_loader, epochs, save_path)
        
        # 测试模型
        test_loss, test_acc, _, _, _, _ = trainer.test(test_loader)
        print(f"测试集准确率: {test_acc:.4f}")
        
        # 保存结果
        results[f'epochs_{epochs}'] = {
            'history': history,
            'test_acc': test_acc,
            'epochs': epochs
        }
    
    # 绘制结果
    plot_epoch_comparison(results)
    
    return results


def plot_lr_comparison(results, epochs):
    """
    绘制学习率比较图表
    
    参数:
        results: 实验结果字典
        epochs: 训练轮次
    """
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 绘制学习率变化
    ax_lr = axes[0, 0]
    for name, result in results.items():
        if 'lr_history' in result['history']:
            ax_lr.plot(result['history']['lr_history'], label=name)
    ax_lr.set_title('学习率变化')
    ax_lr.set_xlabel('轮次')
    ax_lr.set_ylabel('学习率')
    ax_lr.set_yscale('log')
    ax_lr.legend()
    
    # 绘制训练损失
    ax_loss = axes[0, 1]
    for name, result in results.items():
        ax_loss.plot(result['history']['train_loss'], label=f"{name} (训练)")
        # ax_loss.plot(result['history']['val_loss'], '--', label=f"{name} (验证)")
    ax_loss.set_title('损失曲线')
    ax_loss.set_xlabel('轮次')
    ax_loss.set_ylabel('损失')
    ax_loss.legend()
    
    # 绘制训练准确率
    ax_acc = axes[1, 0]
    for name, result in results.items():
        ax_acc.plot(result['history']['train_acc'], label=f"{name} (训练)")
        # ax_acc.plot(result['history']['val_acc'], '--', label=f"{name} (验证)")
    ax_acc.set_title('准确率曲线')
    ax_acc.set_xlabel('轮次')
    ax_acc.set_ylabel('准确率')
    ax_acc.legend()
    
    # 绘制测试准确率
    ax_test = axes[1, 1]
    names = list(results.keys())
    train_accs = [result['history']['train_acc'][-1] for name, result in results.items()]
    val_accs = [result['history']['val_acc'][-1] for name, result in results.items()]
    test_accs = [results[name]['test_acc'] for name in names]
    
    x = np.arange(len(names))
    width = 0.25
    
    ax_test.bar(x - width, train_accs, width, label='训练集')
    ax_test.bar(x, val_accs, width, label='验证集')
    ax_test.bar(x + width, test_accs, width, label='测试集')
    
    ax_test.set_title('不同学习率的准确率对比')
    ax_test.set_xticks(x)
    ax_test.set_xticklabels(names, rotation=45, ha='right')
    ax_test.set_ylabel('准确率')
    ax_test.set_ylim(0.9, 1)
    ax_test.legend()
    
    # 在柱状图上方添加数值标签
    for i, (train, val, test) in enumerate(zip(train_accs, val_accs, test_accs)):
        ax_test.text(i - width, train + 0.001, f'{train:.4f}', ha='center', va='bottom')
        ax_test.text(i, val + 0.001, f'{val:.4f}', ha='center', va='bottom')
        ax_test.text(i + width, test + 0.001, f'{test:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENT_PATH, f'lr_comparison.png'))
    plt.close()


def plot_optimizer_comparison(results, epochs):
    """
    绘制优化器比较图表
    
    参数:
        results: 实验结果字典
        epochs: 训练轮次
    """
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 绘制训练损失
    ax_train_loss = axes[0, 0]
    for name, result in results.items():
        ax_train_loss.plot(result['history']['train_loss'], label=name)
    ax_train_loss.set_title('训练损失')
    ax_train_loss.set_xlabel('轮次')
    ax_train_loss.set_ylabel('损失')
    ax_train_loss.legend()
    
    # 绘制验证损失
    ax_val_loss = axes[0, 1]
    for name, result in results.items():
        ax_val_loss.plot(result['history']['val_loss'], label=name)
    ax_val_loss.set_title('验证损失')
    ax_val_loss.set_xlabel('轮次')
    ax_val_loss.set_ylabel('损失')
    ax_val_loss.legend()
    
    # 绘制训练准确率
    ax_train_acc = axes[1, 0]
    for name, result in results.items():
        ax_train_acc.plot(result['history']['train_acc'], label=name)
    ax_train_acc.set_title('训练准确率')
    ax_train_acc.set_xlabel('轮次')
    ax_train_acc.set_ylabel('准确率')
    ax_train_acc.legend()
    
    # 绘制验证准确率
    ax_val_acc = axes[1, 1]
    for name, result in results.items():
        ax_val_acc.plot(result['history']['val_acc'], label=name)
    ax_val_acc.set_title('验证准确率')
    ax_val_acc.set_xlabel('轮次')
    ax_val_acc.set_ylabel('准确率')
    ax_val_acc.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENT_PATH, f'optimizer_comparison.png'))
    plt.close()
    
    # 绘制测试准确率对比
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    train_accs = [result['history']['train_acc'][-1] for name, result in results.items()]
    val_accs = [result['history']['val_acc'][-1] for name, result in results.items()]
    test_accs = [results[name]['test_acc'] for name in names]
    
    x = np.arange(len(names))
    width = 0.25
    
    plt.bar(x - width, train_accs, width, label='训练集')
    plt.bar(x, val_accs, width, label='验证集')
    plt.bar(x + width, test_accs, width, label='测试集')
    
    plt.title('不同优化器的准确率对比')
    plt.xticks(x, names)
    plt.ylabel('准确率')
    plt.ylim(0.9, 1)
    plt.legend()
    
    # 在柱状图上方添加数值标签
    for i, (train, val, test) in enumerate(zip(train_accs, val_accs, test_accs)):
        plt.text(i - width, train + 0.001, f'{train:.4f}', ha='center', va='bottom')
        plt.text(i, val + 0.001, f'{val:.4f}', ha='center', va='bottom')
        plt.text(i + width, test + 0.001, f'{test:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENT_PATH, f'optimizer_test_acc.png'))
    plt.close()


def plot_epoch_comparison(results):
    """
    绘制训练轮次比较图表
    
    参数:
        results: 实验结果字典
    """
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 绘制训练损失
    ax_train_loss = axes[0, 0]
    max_epochs = max([result['epochs'] for name, result in results.items()])
    for name, result in results.items():
        epochs = result['epochs']
        x = np.arange(epochs)
        ax_train_loss.plot(x, result['history']['train_loss'], label=f"Epochs={epochs}")
    ax_train_loss.set_title('训练损失')
    ax_train_loss.set_xlabel('轮次')
    ax_train_loss.set_ylabel('损失')
    ax_train_loss.set_xlim(0, max_epochs - 1)
    ax_train_loss.legend()
    
    # 绘制验证损失
    ax_val_loss = axes[0, 1]
    for name, result in results.items():
        epochs = result['epochs']
        x = np.arange(epochs)
        ax_val_loss.plot(x, result['history']['val_loss'], label=f"Epochs={epochs}")
    ax_val_loss.set_title('验证损失')
    ax_val_loss.set_xlabel('轮次')
    ax_val_loss.set_ylabel('损失')
    ax_val_loss.set_xlim(0, max_epochs - 1)
    ax_val_loss.legend()
    
    # 绘制训练准确率
    ax_train_acc = axes[1, 0]
    for name, result in results.items():
        epochs = result['epochs']
        x = np.arange(epochs)
        ax_train_acc.plot(x, result['history']['train_acc'], label=f"Epochs={epochs}")
    ax_train_acc.set_title('训练准确率')
    ax_train_acc.set_xlabel('轮次')
    ax_train_acc.set_ylabel('准确率')
    ax_train_acc.set_xlim(0, max_epochs - 1)
    ax_train_acc.legend()
    
    # 绘制验证准确率
    ax_val_acc = axes[1, 1]
    for name, result in results.items():
        epochs = result['epochs']
        x = np.arange(epochs)
        ax_val_acc.plot(x, result['history']['val_acc'], label=f"Epochs={epochs}")
    ax_val_acc.set_title('验证准确率')
    ax_val_acc.set_xlabel('轮次')
    ax_val_acc.set_ylabel('准确率')
    ax_val_acc.set_xlim(0, max_epochs - 1)
    ax_val_acc.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENT_PATH, f'epoch_comparison.png'))
    plt.close()
    
    # 绘制测试准确率对比
    plt.figure(figsize=(10, 6))
    names = [f"Epochs={result['epochs']}" for name, result in results.items()]
    train_accs = [result['history']['train_acc'][-1] for name, result in results.items()]
    val_accs = [result['history']['val_acc'][-1] for name, result in results.items()]
    test_accs = [result['test_acc'] for name, result in results.items()]
    epochs_list = [result['epochs'] for name, result in results.items()]
    
    # 按轮次排序
    sorted_indices = np.argsort(epochs_list)
    names = [names[i] for i in sorted_indices]
    train_accs = [train_accs[i] for i in sorted_indices]
    val_accs = [val_accs[i] for i in sorted_indices]
    test_accs = [test_accs[i] for i in sorted_indices]
    
    x = np.arange(len(names))
    width = 0.25
    
    plt.bar(x - width, train_accs, width, label='训练集')
    plt.bar(x, val_accs, width, label='验证集')
    plt.bar(x + width, test_accs, width, label='测试集')
    
    plt.title('不同训练轮次的准确率对比')
    plt.xticks(x, names)
    plt.ylabel('准确率')
    plt.ylim(0.9, 1)
    plt.legend()
    
    # 在柱状图上方添加数值标签
    for i, (train, val, test) in enumerate(zip(train_accs, val_accs, test_accs)):
        plt.text(i - width, train + 0.001, f'{train:.4f}', ha='center', va='bottom')
        plt.text(i, val + 0.001, f'{val:.4f}', ha='center', va='bottom')
        plt.text(i + width, test + 0.001, f'{test:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENT_PATH, f'epoch_test_acc.png'))
    plt.close()


def main():
    """主函数"""
    # 设置随机种子，确保实验可重复
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 定义实验参数
    dataset = 'mnist'
    epochs = 10
    
    # 运行学习率实验
    lr_results = run_lr_experiment(dataset, epochs)
    
    # 运行优化器实验
    opt_results = run_optimizer_experiment(dataset, epochs)
    
    # 运行训练轮次实验
    epoch_results = run_epoch_experiment(dataset)
    
    print("\n所有实验完成!")
    print(f"结果保存在: {EXPERIMENT_PATH}")


if __name__ == "__main__":
    main() 
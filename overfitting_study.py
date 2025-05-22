"""
过拟合研究
使用ResNet复杂模型研究过拟合问题及其解决方案
1. Dropout层的影响
2. BN层的作用
3. 正则化参数的影响
4. K折交叉验证的作用
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import pickle

from config import SAVE_PATH, FIGURE_PATH, USE_AMP, MODEL_CONFIGS
from models.cnn_model import get_cnn_model
from utils.data_loader import get_dataloaders
from utils.training import Trainer

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建实验结果保存路径
EXPERIMENT_PATH = os.path.join(FIGURE_PATH, 'overfitting_study')
os.makedirs(EXPERIMENT_PATH, exist_ok=True)

# 实验配置
DROPOUT_CONFIGS = [0, 0.1, 0.2, 0.3, 0.4, 0.5 ]
REGULARIZATION_CONFIGS = {
    'L1': [1e-5, 1e-4, 1e-3],
    'L2': [1e-5, 1e-4, 1e-3]
}
K_FOLD_CONFIGS = [3, 5, 10]


class OverfittingTrainer(Trainer):
    """
    过拟合研究训练器
    扩展Trainer类，添加正则化和BN控制功能
    """
    def __init__(self, model, device=None, optimizer_name='SGD', optimizer_params=None,
                 use_bn=False, regularization_type=None, regularization_weight=0):
        """
        初始化过拟合研究训练器
        
        参数:
            model: 要训练的模型
            device: 训练设备
            optimizer_name: 优化器名称
            optimizer_params: 优化器参数
            use_bn: 是否使用BN层
            regularization_type: 正则化类型 ('L1', 'L2', None)
            regularization_weight: 正则化权重
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
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )

        self.use_bn = use_bn
        self.regularization_type = regularization_type
        self.regularization_weight = regularization_weight

    
    def _train_epoch(self, train_loader):
        """
        训练一个轮次
        
        参数:
            train_loader: 训练数据加载器
            
        返回:
            train_loss: 训练损失
            train_acc: 训练准确率
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc='Training'):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 前向传播
            if USE_AMP:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    # 添加正则化损失
                    if self.regularization_type == 'L1':
                        l1_reg = torch.tensor(0., device=self.device)
                        for param in self.model.parameters():
                            l1_reg += torch.norm(param, 1)
                        loss += self.regularization_weight * l1_reg
                    elif self.regularization_type == 'L2':
                        l2_reg = torch.tensor(0., device=self.device)
                        for param in self.model.parameters():
                            l2_reg += torch.norm(param, 2)
                        loss += self.regularization_weight * l2_reg
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 添加正则化损失
                if self.regularization_type == 'L1':
                    l1_reg = torch.tensor(0., device=self.device)
                    for param in self.model.parameters():
                        l1_reg += torch.norm(param, 1)
                    loss += self.regularization_weight * l1_reg
                elif self.regularization_type == 'L2':
                    l2_reg = torch.tensor(0., device=self.device)
                    for param in self.model.parameters():
                        l2_reg += torch.norm(param, 2)
                    loss += self.regularization_weight * l2_reg
            
            # 反向传播
            self.optimizer.zero_grad()
            if USE_AMP:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(train_loader), correct / total


def run_dropout_experiment(dataset='mnist', epochs=10):
    """
    运行Dropout实验
    
    参数:
        dataset: 数据集名称
        epochs: 训练轮次
        
    返回:
        results: 实验结果字典
    """
    print("\n=== 运行Dropout实验 ===")
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(dataset)
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 存储结果
    results = {}
    
    cnn_config = MODEL_CONFIGS['complex']
    # Dropout实验
    for dropout_rate in DROPOUT_CONFIGS:
        print(f"\n--- Dropout率: {dropout_rate} ---")
        
        # 创建模型
        model = get_cnn_model(
            model_type='CNN',
            num_filters=cnn_config['num_filters'],
            num_layers=cnn_config['num_layers'],
            dropout_rate=dropout_rate
        )
        
        # 创建训练器
        trainer = OverfittingTrainer(
            model=model,
            device=device,
            optimizer_name='SGD',
            optimizer_params={'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0005}
        )
        
        # 训练模型
        save_path = os.path.join(SAVE_PATH, f"dropout_{dropout_rate}")
        history = trainer.train(train_loader, val_loader, epochs, save_path)
        
        # 测试模型
        test_loss, test_acc, _, _, _, _ = trainer.test(test_loader)
        print(f"测试集准确率: {test_acc:.4f}")
        
        # 保存结果
        results[f'dropout_{dropout_rate}'] = {
            'history': history,
            'test_acc': test_acc,
            'params': {'dropout_rate': dropout_rate}
        }
    
    # 保存结果
    save_file_path = os.path.join(EXPERIMENT_PATH, 'dropout_results.pkl')
    with open(save_file_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Dropout experiment results saved to {save_file_path}")

    # 绘制结果
    plot_dropout_comparison(results, epochs)
    
    return results


def run_bn_experiment(dataset='mnist', epochs=10):
    """
    运行BN层实验
    
    参数:
        dataset: 数据集名称
        epochs: 训练轮次
        
    返回:
        results: 实验结果字典
    """
    print("\n=== 运行BN层实验 ===")
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(dataset)
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 存储结果
    results = {}
    
    cnn_config = MODEL_CONFIGS['complex']
    # BN实验
    for use_bn in [True, False]:
        print(f"\n--- 使用BN层: {use_bn} ---")
        
        # 创建模型
        model = get_cnn_model(
            model_type='CNN',
            num_filters=cnn_config['num_filters'],
            num_layers=cnn_config['num_layers'],
            use_bn=use_bn
        )
        
        # 创建训练器
        trainer = OverfittingTrainer(
            model=model,
            device=device,
            optimizer_name='SGD',
            optimizer_params={'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0005},
            use_bn=use_bn
        )
        
        # 训练模型
        save_path = os.path.join(SAVE_PATH, f"bn_{use_bn}")
        history = trainer.train(train_loader, val_loader, epochs, save_path)
        
        # 测试模型
        test_loss, test_acc, _, _, _, _ = trainer.test(test_loader)
        print(f"测试集准确率: {test_acc:.4f}")
        
        # 保存结果
        results[f'bn_{use_bn}'] = {
            'history': history,
            'test_acc': test_acc,
            'params': {'use_bn': use_bn}
        }
    
    # 保存结果
    save_file_path = os.path.join(EXPERIMENT_PATH, 'bn_results.pkl')
    with open(save_file_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"BN experiment results saved to {save_file_path}")

    # 绘制结果
    plot_bn_comparison(results, epochs)
    
    return results


def run_regularization_experiment(dataset='mnist', epochs=10):
    """
    运行正则化实验
    
    参数:
        dataset: 数据集名称
        epochs: 训练轮次
        
    返回:
        results: 实验结果字典
    """
    print("\n=== 运行正则化实验 ===")
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_dataloaders(dataset)
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 存储结果
    results = {}
    
    cnn_config = MODEL_CONFIGS['complex']
    # 正则化实验
    for reg_type, weights in REGULARIZATION_CONFIGS.items():
        for weight in weights:
            print(f"\n--- {reg_type}正则化, 权重: {weight} ---")
            
            # 创建模型
            model = get_cnn_model(
                model_type='CNN',
                num_filters=cnn_config['num_filters'],
                num_layers=cnn_config['num_layers']
            )
            
            # 创建训练器
            trainer = OverfittingTrainer(
                model=model,
                device=device,
                optimizer_name='SGD',
                optimizer_params={'lr': 0.001, 'momentum': 0.9},
                regularization_type=reg_type,
                regularization_weight=weight
            )
            
            # 训练模型
            save_path = os.path.join(SAVE_PATH, f"reg_{reg_type}_{weight}")
            history = trainer.train(train_loader, val_loader, epochs, save_path)
            
            # 测试模型
            test_loss, test_acc, _, _, _, _ = trainer.test(test_loader)
            print(f"测试集准确率: {test_acc:.4f}")
            
            # 保存结果
            results[f'reg_{reg_type}_{weight}'] = {
                'history': history,
                'test_acc': test_acc,
                'params': {'type': reg_type, 'weight': weight}
            }
    
    # 保存结果
    save_file_path = os.path.join(EXPERIMENT_PATH, 'regularization_results.pkl')
    with open(save_file_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Regularization experiment results saved to {save_file_path}")

    # 绘制结果
    plot_regularization_comparison(results, epochs)
    
    return results


def run_kfold_experiment(dataset='mnist', epochs=10):
    """
    运行K折交叉验证实验
    
    参数:
        dataset: 数据集名称
        epochs: 训练轮次
        
    返回:
        results: 实验结果字典
    """
    print("\n=== 运行K折交叉验证实验 ===")
    
    # 获取数据集
    train_loader, val_loader, test_loader = get_dataloaders(dataset)
    dataset = train_loader.dataset
    
    # 获取设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 存储结果
    results = {}
    
    cnn_config = MODEL_CONFIGS['complex']
    # K折交叉验证实验
    for k in K_FOLD_CONFIGS:
        print(f"\n--- K折交叉验证: K={k} ---")
        
        # 创建K折交叉验证
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        
        # 存储每折的结果
        fold_results = []
        
        # 进行K折交叉验证
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"\n训练第 {fold+1}/{k} 折")
            
            # 创建数据加载器
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(
                dataset,
                batch_size=train_loader.batch_size,
                sampler=train_sampler
            )
            
            val_loader = DataLoader(
                dataset,
                batch_size=val_loader.batch_size,
                sampler=val_sampler
            )
            
            # 创建模型
            model = get_cnn_model(
                model_type='CNN',
                num_filters=cnn_config['num_filters'],
                num_layers=cnn_config['num_layers']
            )
            
            # 创建训练器
            trainer = OverfittingTrainer(
                model=model,
                device=device,
                optimizer_name='SGD',
                optimizer_params={'lr': 0.001, 'momentum': 0.9, 'weight_decay': 0.0005}
            )
            
            # 训练模型
            save_path = os.path.join(SAVE_PATH, f"kfold_{k}_fold_{fold}")
            history = trainer.train(train_loader, val_loader, epochs, save_path)
            
            # 记录结果
            fold_results.append({
                'history': history,
                'val_acc': history['val_acc'][-1]
            })
        
        # 计算平均结果
        avg_val_acc = np.mean([result['val_acc'] for result in fold_results])
        print(f"K={k} 的平均验证准确率: {avg_val_acc:.4f}")
        
        # 保存结果
        results[f'kfold_{k}'] = {
            'fold_results': fold_results,
            'avg_val_acc': avg_val_acc,
            'params': {'k': k}
        }
    
    # 保存整体K折结果 (在K折循环外部)
    save_file_path = os.path.join(EXPERIMENT_PATH, 'kfold_results.pkl')
    with open(save_file_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"K-Fold experiment results saved to {save_file_path}")

    # 绘制结果
    plot_kfold_comparison(results, epochs)
    
    return results


def plot_dropout_comparison(results, epochs):
    """
    绘制Dropout比较图表
    
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
    plt.savefig(os.path.join(EXPERIMENT_PATH, f'dropout_comparison.png'))
    plt.close()
    
    # 绘制准确率对比直方图
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
    
    plt.title('不同Dropout率的准确率对比')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('准确率')
    plt.ylim(0.9, 1)
    plt.legend()
    
    # 在柱状图上方添加数值标签
    for i, (train, val, test) in enumerate(zip(train_accs, val_accs, test_accs)):
        plt.text(i - width, train + 0.001, f'{train:.4f}', ha='center', va='bottom')
        plt.text(i, val + 0.001, f'{val:.4f}', ha='center', va='bottom')
        plt.text(i + width, test + 0.001, f'{test:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENT_PATH, f'dropout_test_acc.png'))
    plt.close()


def plot_bn_comparison(results, epochs):
    """
    绘制BN层比较图表
    
    参数:
        results: 实验结果字典
        epochs: 训练轮次
    """
    # --- 绘制损失曲线 (现有代码) ---
    for bn_setting in ['bn_True', 'bn_False']:
        if bn_setting in results: # 确保结果存在
            result = results[bn_setting]

            plt.figure(figsize=(10, 6))
            plt.plot(result['history']['train_loss'], label='训练损失')
            plt.plot(result['history']['val_loss'], label='验证损失')
            plt.title(f'{"有" if bn_setting == "bn_True" else "无"}BN层的损失曲线')
            plt.xlabel('轮次')
            plt.ylabel('损失')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            save_path_loss = os.path.join(EXPERIMENT_PATH, f'bn_loss_{bn_setting}.png')
            plt.savefig(save_path_loss)
            print(f"BN loss plot saved to {save_path_loss}")
            plt.close()
        else:
             print(f"Warning: Results for {bn_setting} not found. Skipping loss plot.")


    # --- 绘制准确率对比直方图 (新增代码) ---
    plt.figure(figsize=(10, 6))
    names = []
    train_accs = []
    val_accs = []
    test_accs = []

    # 确保顺序一致
    for bn_setting in ['bn_True', 'bn_False']:
        if bn_setting in results:
            name = '有BN层' if bn_setting == 'bn_True' else '无BN层'
            names.append(name)
            train_accs.append(results[bn_setting]['history']['train_acc'][-1])
            val_accs.append(results[bn_setting]['history']['val_acc'][-1])
            test_accs.append(results[bn_setting]['test_acc'])
        else:
             print(f"Warning: Results for {bn_setting} not found. Skipping for accuracy plot.")


    if names: # 只有当有数据时才绘制
        x = np.arange(len(names))
        width = 0.25

        plt.bar(x - width, train_accs, width, label='训练集')
        plt.bar(x, val_accs, width, label='验证集')
        plt.bar(x + width, test_accs, width, label='测试集')

        plt.title('有无BN层的准确率对比')
        plt.xticks(x + width/2, names) # 将x轴标签放在中间
        plt.ylabel('准确率')
        # 避免ymin为0导致图表压缩，根据最小值动态调整
        min_acc = min(min(train_accs), min(val_accs), min(test_accs))
        plt.ylim(max(0.0, min_acc * 0.95), 1.0) # 保证不小于0，且上限为1
        plt.legend()

        # 在柱状图上方添加数值标签
        for i, (train, val, test) in enumerate(zip(train_accs, val_accs, test_accs)):
            plt.text(i - width, train + 0.005, f'{train:.4f}', ha='center', va='bottom', fontsize=8)
            plt.text(i, val + 0.005, f'{val:.4f}', ha='center', va='bottom', fontsize=8)
            plt.text(i + width, test + 0.005, f'{test:.4f}', ha='center', va='bottom', fontsize=8)


        plt.tight_layout()
        save_path_acc = os.path.join(EXPERIMENT_PATH, 'bn_accuracy_comparison.png')
        plt.savefig(save_path_acc)
        print(f"BN accuracy plot saved to {save_path_acc}")
        plt.close()
    else:
        print("No complete BN experiment results available for accuracy plot.")


def plot_regularization_comparison(results, epochs):
    """
    绘制正则化比较图表
    
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
    plt.savefig(os.path.join(EXPERIMENT_PATH, f'regularization_comparison.png'))
    plt.close()
    
    # 绘制准确率对比直方图
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
    
    plt.title('不同正则化方法的准确率对比')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('准确率')
    plt.ylim(0.9, 1)
    plt.legend()
    
    # 在柱状图上方添加数值标签
    for i, (train, val, test) in enumerate(zip(train_accs, val_accs, test_accs)):
        plt.text(i - width, train + 0.001, f'{train:.4f}', ha='center', va='bottom')
        plt.text(i, val + 0.001, f'{val:.4f}', ha='center', va='bottom')
        plt.text(i + width, test + 0.001, f'{test:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENT_PATH, f'regularization_test_acc.png'))
    plt.close()


def plot_kfold_comparison(results, epochs):
    """
    绘制K折交叉验证比较图表
    
    参数:
        results: 实验结果字典
        epochs: 训练轮次
    """
    # 为每个K值绘制损失曲线
    for k_name, result in results.items():
        plt.figure(figsize=(10, 6))
        
        # 计算每个折的平均损失
        avg_train_loss = np.mean([fold['history']['train_loss'] for fold in result['fold_results']], axis=0)
        avg_val_loss = np.mean([fold['history']['val_loss'] for fold in result['fold_results']], axis=0)
        
        plt.plot(avg_train_loss, label='训练损失')
        plt.plot(avg_val_loss, label='验证损失')
        plt.title(f'K={result["params"]["k"]} 的平均损失曲线')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(EXPERIMENT_PATH, f'kfold_loss_{k_name}.png'))
        plt.close()
    

    # 绘制不同K值的折间方差
    k_values = []
    variances = []

    for name, result in results.items():
        k = result['params']['k']
        val_accs = [fold['val_acc'] for fold in result['fold_results']]
        variance = np.var(val_accs)
        
        k_values.append(k)
        variances.append(variance)

    plt.plot(k_values, variances, 'o-')
    plt.title('不同K值的折间方差')
    plt.xlabel('K值')
    plt.ylabel('方差')
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENT_PATH, f'kfold_variance.png'))
    plt.close()


    # 绘制准确率对比直方图
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    k_values = [result['params']['k'] for result in results.values()]
    
    # 计算每个K值的平均准确率
    train_accs = []
    val_accs = []
    test_accs = []
    
    for result in results.values():
        train_accs.append(np.mean([fold['history']['train_acc'][-1] for fold in result['fold_results']]))
        val_accs.append(np.mean([fold['history']['val_acc'][-1] for fold in result['fold_results']]))
        test_accs.append(result['avg_val_acc'])
    
    x = np.arange(len(names))
    width = 0.25
    
    plt.bar(x - width, train_accs, width, label='训练集')
    plt.bar(x, val_accs, width, label='验证集')
    plt.bar(x + width, test_accs, width, label='测试集')
    
    plt.title('不同K值的准确率对比')
    plt.xticks(x, [f'K={k}' for k in k_values])
    plt.ylabel('准确率')
    plt.ylim(0.9, 1)
    plt.legend()
    
    # 在柱状图上方添加数值标签
    for i, (train, val, test) in enumerate(zip(train_accs, val_accs, test_accs)):
        plt.text(i - width, train + 0.001, f'{train:.4f}', ha='center', va='bottom')
        plt.text(i, val + 0.001, f'{val:.4f}', ha='center', va='bottom')
        plt.text(i + width, test + 0.001, f'{test:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENT_PATH, f'kfold_accuracy.png'))
    plt.close()


def main():
    """主函数"""
    # 设置随机种子，确保实验可重复
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 定义实验参数
    dataset = 'mnist'
    epochs = 10
    
    # 运行Dropout实验
    dropout_results = run_dropout_experiment(dataset, epochs)
    
    # 运行BN层实验
    # bn_results = run_bn_experiment(dataset, epochs)
    
    # 运行正则化实验
    # reg_results = run_regularization_experiment(dataset, epochs)
    
    # 运行K折交叉验证实验
    # kfold_results = run_kfold_experiment(dataset, epochs)
    
    print("\n所有实验完成!")
    print(f"结果保存在: {EXPERIMENT_PATH}")


if __name__ == "__main__":
    main() 
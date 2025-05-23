import os
import torch
import numpy as np
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pickle
import json
from datetime import datetime

from config import BATCH_SIZE, NUM_WORKERS, FIGURE_PATH, SAVE_PATH, CLASS_NAMES
from models.cnn_model import get_cnn_model
from utils.training import Trainer
from utils.visualization import plot_learning_curves, plot_confusion_matrix, visualize_feature_maps

# ========== 新增：统一结果保存目录 ==========
RESULTS_DIR = os.path.join(FIGURE_PATH, 'cnn_complex_kfold')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 参数配置
NUM_EPOCHS = 15
NUM_FOLDS = 3
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
STEP_SIZE = 5
GAMMA = 0.1
DROPOUT = 0.3
USE_BN = True
NUM_FILTERS = 64
NUM_LAYERS = 4
SEED = 42

# ========== 新增：高斯噪声增强 ==========
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

def get_augmented_mnist_loaders(train_idx, val_idx):
    train_transform = transforms.Compose([
        # transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.1),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # 加载完整MNIST
    full_train = datasets.MNIST('./data', train=True, download=True, transform=None)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=test_transform)
    # 划分训练/验证
    train_subset = Subset(full_train, train_idx)
    val_subset = Subset(full_train, val_idx)
    # 替换transform
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = test_transform
    # DataLoader
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader, test_loader

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SAVE_PATH, exist_ok=True)
    # 3折交叉验证
    full_train = datasets.MNIST('./data', train=True, download=True)
    indices = np.arange(len(full_train))
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    all_val_acc = []
    all_test_acc = []
    all_histories = []
    all_conf_matrices = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ========== 新增：用于多折曲线对比 ==========
    all_train_losses, all_val_losses, all_train_accs, all_val_accs = [], [], [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n===== Fold {fold+1}/{NUM_FOLDS} =====")
        train_loader, val_loader, test_loader = get_augmented_mnist_loaders(train_idx, val_idx)
        model = get_cnn_model(
            model_type='CNN',
            num_filters=NUM_FILTERS,
            num_layers=NUM_LAYERS,
            dropout_rate=DROPOUT,
            use_bn=USE_BN,
            num_classes=10
        )
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
        trainer = Trainer(model, device=device, learning_rate=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        # 替换调度器为StepLR
        trainer.optimizer = optimizer
        trainer.scheduler = scheduler
        save_prefix = os.path.join(SAVE_PATH, f'kfold_{NUM_FOLDS}_fold_{fold}')
        history = trainer.train(train_loader, val_loader, NUM_EPOCHS, save_prefix)
        all_histories.append(history)
        all_train_losses.append(history['train_loss'])
        all_val_losses.append(history['val_loss'])
        all_train_accs.append(history['train_acc'])
        all_val_accs.append(history['val_acc'])
        best_val_acc = max(history['val_acc'])
        all_val_acc.append(best_val_acc)
        # 测试集评估
        test_loss, test_acc, all_preds, all_labels, test_losses, test_accs = trainer.test(test_loader)
        all_test_acc.append(test_acc)
        print(f"Fold {fold+1} Test Accuracy: {test_acc:.4f}")
        # 混淆矩阵和分类报告
        cm = confusion_matrix(all_labels, all_preds)
        all_conf_matrices.append(cm)
        
        # 打印分类报告
        print("\n分类报告:")
        report = classification_report(all_labels, all_preds)
        print(report)
        
        # 保存混淆矩阵和分类报告到文件
        report_path = os.path.join(RESULTS_DIR, f'kfold_{NUM_FOLDS}_fold_{fold}_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Fold {fold+1} 混淆矩阵:\n")
            f.write(str(cm))
            f.write("\n\n分类报告:\n")
            f.write(report)
        
        # 可视化混淆矩阵
        plot_confusion_matrix(cm, CLASS_NAMES)
        plt_path = os.path.join(RESULTS_DIR, f'kfold_{NUM_FOLDS}_fold_{fold}_confusion_matrix.png')
        plt.savefig(plt_path)
        plt.close()
        # 学习曲线
        plot_learning_curves(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'],
                            title=f'Fold {fold+1} Learning Curves',
                            save_path=os.path.join(RESULTS_DIR, f'kfold_{NUM_FOLDS}_fold_{fold}_curves.png'))
    # ========== 新增：多折曲线对比图 ==========
    epochs = range(1, NUM_EPOCHS+1)
    plt.figure(figsize=(12, 5))
    for i in range(NUM_FOLDS):
        plt.plot(epochs, all_train_losses[i], label=f'Train Fold {i+1}')
    plt.title('Train Loss Curves (All Folds)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'all_folds_train_loss.png'))
    plt.close()
    plt.figure(figsize=(12, 5))
    for i in range(NUM_FOLDS):
        plt.plot(epochs, all_val_losses[i], label=f'Val Fold {i+1}')
    plt.title('Val Loss Curves (All Folds)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'all_folds_val_loss.png'))
    plt.close()
    plt.figure(figsize=(12, 5))
    for i in range(NUM_FOLDS):
        plt.plot(epochs, all_train_accs[i], label=f'Train Fold {i+1}')
    plt.title('Train Accuracy Curves (All Folds)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'all_folds_train_acc.png'))
    plt.close()
    plt.figure(figsize=(12, 5))
    for i in range(NUM_FOLDS):
        plt.plot(epochs, all_val_accs[i], label=f'Val Fold {i+1}')
    plt.title('Val Accuracy Curves (All Folds)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'all_folds_val_acc.png'))
    plt.close()
    # ========== 新增：评估报告 ==========
    mean_val = np.mean(all_val_acc)
    std_val = np.std(all_val_acc)
    mean_test = np.mean(all_test_acc)
    std_test = np.std(all_test_acc)
    report_lines = []
    report_lines.append('===== K折交叉验证评估报告 =====\n')
    for i in range(NUM_FOLDS):
        report_lines.append(f'Fold {i+1}: Val Acc = {all_val_acc[i]:.4f}, Test Acc = {all_test_acc[i]:.4f}')
    report_lines.append(f'\nVal Acc Mean = {mean_val:.4f}, Std = {std_val:.4f}')
    report_lines.append(f'Test Acc Mean = {mean_test:.4f}, Std = {std_test:.4f}')
    print('\n'.join(report_lines))
    with open(os.path.join(RESULTS_DIR, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    # 汇总混淆矩阵
    mean_cm = np.mean(np.stack(all_conf_matrices), axis=0)
    plot_confusion_matrix(mean_cm, CLASS_NAMES)
    plt.savefig(os.path.join(RESULTS_DIR, f'kfold_{NUM_FOLDS}_mean_confusion_matrix.png'))
    plt.close()
    # ========== 新增：K折对比图（双y轴） ==========
    fold_ids = np.arange(1, NUM_FOLDS+1)
    last_train_acc = [all_train_accs[i][-1] for i in range(NUM_FOLDS)]
    last_val_acc = [all_val_accs[i][-1] if 'all_val_accs' in locals() else all_val_acc[i] for i in range(NUM_FOLDS)]
    last_test_acc = [all_test_acc[i] for i in range(NUM_FOLDS)]
    last_train_loss = [all_train_losses[i][-1] for i in range(NUM_FOLDS)]
    last_val_loss = [all_val_losses[i][-1] for i in range(NUM_FOLDS)]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    # 准确率
    l1 = ax1.plot(fold_ids, last_train_acc, 'o-', color='#1f77b4', label='Train Acc', markersize=8, linewidth=2)
    l2 = ax1.plot(fold_ids, last_val_acc, 's-', color='#2ca02c', label='Val Acc', markersize=8, linewidth=2)
    l3 = ax1.plot(fold_ids, last_test_acc, '^--', color='#ff7f0e', label='Test Acc', markersize=8, linewidth=2)
    # 损失
    l4 = ax2.plot(fold_ids, last_train_loss, 'o--', color='#9467bd', label='Train Loss', markersize=8, linewidth=2)
    l5 = ax2.plot(fold_ids, last_val_loss, 's--', color='#8c564b', label='Val Loss', markersize=8, linewidth=2)
    # 美化
    ax1.set_xlabel('Fold', fontsize=13)
    ax1.set_ylabel('准确率', fontsize=13, color='#1f77b4')
    ax2.set_ylabel('损失', fontsize=13, color='#8c564b')
    ax1.set_xticks(fold_ids)
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, max(last_train_loss + last_val_loss)*1.1)
    ax1.grid(True, linestyle='--', alpha=0.5)
    # 标注每个点
    for x, y in zip(fold_ids, last_train_acc):
        ax1.text(x-0.08, y+0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=10, color='#1f77b4')
    for x, y in zip(fold_ids, last_val_acc):
        ax1.text(x+0.08, y+0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=10, color='#2ca02c')
    for x, y in zip(fold_ids, last_test_acc):
        ax1.text(x, y-0.04, f'{y:.3f}', ha='center', va='top', fontsize=10, color='#ff7f0e')
    for x, y in zip(fold_ids, last_train_loss):
        ax2.text(x-0.08, y+0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=10, color='#9467bd')
    for x, y in zip(fold_ids, last_val_loss):
        ax2.text(x+0.08, y+0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=10, color='#8c564b')
    # 图例
    lines = l1 + l2 + l3 + l4 + l5
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', fontsize=12, ncol=3, frameon=True)
    plt.title('K折交叉验证结果对比（训练/验证/测试）', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'kfold_comparison.png'))
    plt.close()

if __name__ == '__main__':
    main() 
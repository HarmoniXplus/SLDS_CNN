import os
import torch
import numpy as np
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import pickle
import json
from datetime import datetime
import argparse
import sys

from config import BATCH_SIZE, NUM_WORKERS, FIGURE_PATH, SAVE_PATH, CLASS_NAMES
from models.cnn_model import get_cnn_model
from utils.training import Trainer
from utils.visualization import plot_learning_curves, plot_confusion_matrix, visualize_feature_maps

def parse_args():
    parser = argparse.ArgumentParser(description='CNN/ResNet模型训练和评估')
    parser.add_argument('--model_type', type=str, default='CNN', choices=['CNN', 'resnet'],
                      help='模型类型: CNN 或 resnet')
    return parser.parse_args()

# ========== 新增：统一结果保存目录 ==========
def get_results_dir(model_type):
    return os.path.join(FIGURE_PATH, f'{model_type}_complex_kfold')

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

def calculate_metrics(labels, predictions, class_names):
    """计算详细的评估指标"""
    metrics = {}
    # 计算混淆矩阵
    metrics['confusion_matrix'] = confusion_matrix(labels, predictions)
    # 计算准确率
    metrics['accuracy'] = accuracy_score(labels, predictions)
    # 计算精确率、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    metrics['precision_weighted'] = precision
    metrics['recall_weighted'] = recall
    metrics['f1_weighted'] = f1
    return metrics

def main():
    # 解析命令行参数
    args = parse_args()
    model_type = args.model_type
    
    # 设置结果保存目录
    RESULTS_DIR = get_results_dir(model_type)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)
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
        
        # 根据模型类型创建不同的模型
        if model_type == 'CNN':
            model = get_cnn_model(
                model_type='CNN',
                num_filters=NUM_FILTERS,
                num_layers=NUM_LAYERS,
                dropout_rate=DROPOUT,
                use_bn=USE_BN,
                num_classes=10
            )
        else:  # resnet
            model = get_cnn_model(
                model_type='resnet',
                num_blocks=[3, 4, 6],  # ResNet-complex结构
                dropout_rate=DROPOUT,
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
        
        # 计算详细评估指标
        metrics = calculate_metrics(all_labels, all_preds, CLASS_NAMES)
        
        # 打印评估指标
        print(f"测试集准确率: {metrics['accuracy']:.4f}")
        print(f"测试集精确率: {metrics['precision_weighted']:.4f}")
        print(f"测试集召回率: {metrics['recall_weighted']:.4f}")
        print(f"测试集F1分数: {metrics['f1_weighted']:.4f}")
        
        # 混淆矩阵
        cm = metrics['confusion_matrix']
        all_conf_matrices.append(cm)
        
        # 可视化混淆矩阵
        plot_confusion_matrix(cm, CLASS_NAMES)
        plt_path = os.path.join(RESULTS_DIR, f'kfold_{NUM_FOLDS}_fold_{fold}_confusion_matrix.png')
        plt.savefig(plt_path)
        plt.close()
        
        # 保存评估结果为JSON
        results = {
            'fold': fold + 1,
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision_weighted']),
            'recall': float(metrics['recall_weighted']),
            'f1': float(metrics['f1_weighted'])
        }
        
        results_path = os.path.join(RESULTS_DIR, f'kfold_{NUM_FOLDS}_fold_{fold}_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
            
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


if __name__ == '__main__':
    main() 
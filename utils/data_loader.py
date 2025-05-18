"""
数据加载模块
包含数据加载和预处理函数
"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

from config import (
    BATCH_SIZE, NUM_WORKERS, VAL_SIZE,
    DATA_PATH, DDR_DATA_PATH, DDR_TRAIN_LABELS,
    DDR_VAL_LABELS, DDR_TEST_LABELS, PIN_MEMORY
)


def get_dataloaders(dataset='mnist'):
    """
    获取数据加载器
    
    参数:
        dataset: 数据集名称 ('mnist' 或 'ddr')
    
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    if dataset == 'mnist':
        return get_mnist_dataloaders()
    elif dataset == 'ddr':
        return get_ddr_dataloaders()
    else:
        raise ValueError(f"不支持的数据集: {dataset}")


def get_mnist_dataloaders():
    """
    获取MNIST数据加载器
    
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载训练集
    train_dataset = datasets.MNIST(
        DATA_PATH, train=True, download=True,
        transform=transform
    )
    
    # 划分训练集和验证集
    train_size = len(train_dataset) - VAL_SIZE
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, VAL_SIZE]
    )
    
    # 加载测试集
    test_dataset = datasets.MNIST(
        DATA_PATH, train=False,
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    return train_loader, val_loader, test_loader


class DDRDataset(Dataset):
    """DDR数据集类"""
    def __init__(self, image_paths, labels, transform=None):
        """
        初始化DDR数据集
        
        参数:
            image_paths: 图像路径列表
            labels: 标签列表
            transform: 数据预处理
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # 应用预处理
        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        label = self.labels[idx]
        
        return image, label


def load_ddr_data(label_file):
    """
    加载DDR数据集
    
    参数:
        label_file: 标签文件路径
    
    返回:
        image_paths: 图像路径列表
        labels: 标签列表
    """
    image_paths = []
    labels = []
    
    with open(label_file, 'r') as f:
        for line in f:
            # 解析每行数据
            image_path, label = line.strip().split()
            image_path = os.path.join(DDR_DATA_PATH, image_path)
            label = int(label)
            
            image_paths.append(image_path)
            labels.append(label)
    
    return image_paths, labels


def get_ddr_dataloaders():
    """
    获取DDR数据加载器
    
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    # 加载数据
    train_paths, train_labels = load_ddr_data(DDR_TRAIN_LABELS)
    val_paths, val_labels = load_ddr_data(DDR_VAL_LABELS)
    test_paths, test_labels = load_ddr_data(DDR_TEST_LABELS)
    
    # 创建数据集
    train_dataset = DDRDataset(train_paths, train_labels)
    val_dataset = DDRDataset(val_paths, val_labels)
    test_dataset = DDRDataset(test_paths, test_labels)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    return train_loader, val_loader, test_loader


def get_sample_images(dataloader, num_samples=1):
    """获取样本图像和标签"""
    images, labels = next(iter(dataloader))
    return images[:num_samples], labels[:num_samples] 
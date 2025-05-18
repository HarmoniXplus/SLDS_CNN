"""
训练工具模块
包含训练、验证和测试函数
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import SAVE_PATH, FIGURE_PATH, USE_AMP, USE_MULTI_GPU
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler


class Trainer:
    """
    模型训练器
    
    封装训练、验证和测试流程
    """
    def __init__(self, model, device=None, learning_rate=0.001, momentum=0.9, weight_decay=1e-4):
        """
        初始化训练器
        
        参数:
            model: 要训练的模型
            device: 训练设备，如不提供则自动选择
            learning_rate: 学习率
            momentum: 动量
            weight_decay: 权重衰减
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 多GPU支持
        if USE_MULTI_GPU and torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU训练")
            self.model = nn.DataParallel(model)
        else:
            self.model = model
            
        self.model.to(self.device)
        
        # 创建优化器
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        #可使用ADM优化器
        
        # 创建学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        # self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        #可使用StepLR学习率调度器
        #可使用ReduceLROnPlateau学习率调度器
        #可使用CosineAnnealingLR学习率调度器
        #可使用CyclicLR学习率调度器
        #可使用OneCycleLR学习率调度器
        #余弦退火学习率调度器
        
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
        
        # 确保模型保存路径存在
        os.makedirs(SAVE_PATH, exist_ok=True)
    
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
            # 训练一个轮次
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self._validate(val_loader)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录训练历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # 打印训练信息
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(f"{save_path}_best.pth")
            
            # 保存最后一轮模型
            if epoch == epochs - 1:
                self.save_model(f"{save_path}_last.pth")
        
        # 绘制学习曲线
        self._plot_learning_curves(save_path)
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_acc': self.train_accs,
            'val_acc': self.val_accs
        }
    
    def _train_epoch(self, train_loader):
        """训练一个轮次"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc='Training'):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 使用混合精度训练
            if USE_AMP:
                with autocast():
                    # 前向传播
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(train_loader), correct / total
    
    def _validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc='Validating'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 使用混合精度验证
                if USE_AMP:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # 统计
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(val_loader), correct / total
    
    def test(self, test_loader):
        """
        测试模型
        
        参数:
            test_loader: 测试数据加载器
        
        返回:
            test_loss: 测试损失
            test_acc: 测试准确率
            all_preds: 所有预测结果
            all_labels: 所有真实标签
            test_losses: 每个批次的测试损失
            test_accs: 每个批次的测试准确率
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        test_losses = []
        test_accs = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc='Testing'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 使用混合精度测试
                if USE_AMP:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # 统计
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 收集预测结果
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                
                # 记录每个批次的损失和准确率
                test_losses.append(loss.item())
                test_accs.append(predicted.eq(targets).sum().item() / targets.size(0))
        
        return total_loss / len(test_loader), correct / total, all_preds, all_labels, test_losses, test_accs
    
    def save_model(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """加载模型"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
    
    def _plot_learning_curves(self, save_path):
        """绘制学习曲线"""
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制损失曲线
        ax1.plot(self.train_losses, label='Train')
        ax1.plot(self.val_losses, label='Validation')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # 绘制准确率曲线
        ax2.plot(self.train_accs, label='Train')
        ax2.plot(self.val_accs, label='Validation')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(f"{save_path}_curves.png")
        plt.close()


def train_multiple_models(models, trainloader, valloader, epochs=10, **kwargs):
    """
    训练多个模型并比较它们的性能
    
    参数:
        models: 模型字典，键为模型名称，值为模型实例
        trainloader: 训练数据加载器
        valloader: 验证数据加载器
        epochs: 训练轮次
        **kwargs: 传递给Trainer的其他参数
        
    返回:
        results: 包含所有模型训练历史的字典
    """
    results = {}
    
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        trainer = Trainer(model, **kwargs)
        history = trainer.train(trainloader, valloader, epochs, os.path.join(SAVE_PATH, name))
        results[name] = {
            'model': model,
            'trainer': trainer,
            'history': history
        }
    
    return results 
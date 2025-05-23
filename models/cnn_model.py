"""
CNN模型定义模块
包含不同深度和结构的CNN模型实现
"""
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    可配置的简单CNN模型
    适用于MNIST数据集的手写数字识别
    
    参数:
        num_classes: 分类类别数量
        num_filters: 初始卷积层过滤器数量
        num_layers: 卷积层数量
        dropout_rate: Dropout比率
        use_bn: 是否使用BN层
    """
    def __init__(self, num_classes=10, num_filters=16, num_layers=2, dropout_rate=0.0, use_bn=False):
        super().__init__()
        layers = []
        # MNIST是单通道灰度图
        in_channels = 1
        # 构建卷积层
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(dropout_rate))
            in_channels = num_filters
            num_filters *= 2
        
        self.features = nn.Sequential(*layers)
        
        # 计算全连接层的输入特征数量
        # MNIST图像大小为28x28，每经过一次池化层尺寸减半
        feature_size = 28 // (2 ** num_layers)
        fc_input_size = in_channels * feature_size * feature_size
        
        classifier_layers = [
            nn.Flatten(),
            nn.Linear(fc_input_size, 128),
            nn.ReLU()
        ]
        if dropout_rate > 0:
            classifier_layers.append(nn.Dropout(dropout_rate))
        classifier_layers.append(nn.Linear(128, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MLP(nn.Module):
    """
    简单的多层感知机模型
    适用于MNIST数据集的手写数字识别
    
    参数:
        num_classes: 分类类别数量
        hidden_sizes: 隐藏层大小列表，例如[128, 64]表示两个隐藏层
        dropout_rate: Dropout比率，用于防止过拟合
    """
    def __init__(self, num_classes=10, hidden_sizes=[128, 64], dropout_rate=0.5):
        super().__init__()
        
        # 创建网络层
        layers = []
        
        # 输入层 - MNIST图像大小为28x28=784
        input_size = 28 * 28
        
        # 构建隐藏层
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))
        
        # 将所有层组合为序列模型
        self.model = nn.Sequential(
            nn.Flatten(),  # 将图像展平为向量
            *layers
        )
    
    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    """
    残差块实现，用于构建更深层次的CNN
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # 如果输入输出通道数不匹配，添加1x1卷积进行转换
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    基于残差网络的深层CNN实现
    """
    def __init__(self, num_blocks, num_classes=10, dropout_rate=0.0):
        super().__init__()
        self.in_channels = 16
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        # 构建残差网络层
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2, dropout_rate=dropout_rate)
        
        # 全局平均池化和分类器
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride, dropout_rate=0.0):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride, dropout_rate))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def get_cnn_model(model_type='CNN', num_filters=32, num_layers=3, dropout_rate=0.0, use_bn=False, **kwargs):
    """
    工厂函数，根据指定类型创建相应的CNN模型

    参数:
        model_type: 模型类型 ('CNN', 'resnet', 或 'mlp')
        num_filters: 卷积核数量（仅CNN）
        num_layers: 卷积层数（仅CNN）
        dropout_rate: Dropout比率（仅CNN/MLP）
        use_bn: 是否使用BN层（仅CNN）
        **kwargs: 传递给模型构造函数的其他参数
    """
    if model_type == 'CNN':
        return SimpleCNN(num_filters=num_filters, num_layers=num_layers, dropout_rate=dropout_rate, use_bn=use_bn, num_classes=kwargs.get('num_classes', 10))
    elif model_type == 'resnet':
        num_blocks = kwargs.get('num_blocks', [2, 2, 2])
        return ResNet(num_blocks, kwargs.get('num_classes', 10), dropout_rate=dropout_rate)
    elif model_type == 'mlp':
        hidden_sizes = kwargs.get('hidden_sizes', [128, 64])
        dropout_rate = kwargs.get('dropout_rate', 0.5)
        return MLP(
            num_classes=kwargs.get('num_classes', 10),
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}") 
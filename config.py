"""
配置文件
包含模型、训练和数据集相关的配置参数
"""
import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data')
MODEL_PATH = os.path.join(BASE_DIR, 'models')
FIGURE_PATH = os.path.join(BASE_DIR, 'figures')

# DDR数据集路径
DDR_DATA_PATH = os.path.join(DATA_PATH, 'DDR-dataset')
DDR_TRAIN_LABELS = os.path.join(DDR_DATA_PATH, 'train.txt')
DDR_VAL_LABELS = os.path.join(DDR_DATA_PATH, 'val.txt')
DDR_TEST_LABELS = os.path.join(DDR_DATA_PATH, 'test.txt')

# 确保目录存在
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(FIGURE_PATH, exist_ok=True)

# 数据集配置
BATCH_SIZE = 64
NUM_WORKERS = 4
VAL_SIZE = 5000

# 类别名称
MNIST_CLASS_NAMES = [str(i) for i in range(10)]
DDR_CLASS_NAMES = [
    '无DR病变',
    '轻度DR病变',
    '中度DR病变',
    '重度DR病变',
    '增值性DR病变',
    '无法分级'
]

# 模型配置
MODEL_CONFIGS = {
    'simple': {
        'num_filters': 16,    # 初始卷积层的滤波器数量
        'num_layers': 2,      # 卷积层数量
    },
    'medium': {
        'num_filters': 32,
        'num_layers': 3,
    },
    'complex': {
        'num_filters': 64,
        'num_layers': 4,
    }
}

# MLP模型参数
MLP_CONFIGS = {
    'simple': {
        'hidden_sizes': [128],           # 单隐藏层
        'dropout_rate': 0.3,
    },
    'medium': {
        'hidden_sizes': [256, 128],      # 两个隐藏层
        'dropout_rate': 0.5,
    },
    'complex': {
        'hidden_sizes': [512, 256, 128], # 三个隐藏层
        'dropout_rate': 0.5,
    }
}

# 训练配置
LEARNING_RATE = 0.001
EPOCHS = 10
SAVE_PATH = './checkpoints'

# 可视化参数
SAVE_FIGURES = True
FIGURE_PATH = './figures'

# 测试参数
MODEL_PATH = './checkpoints/best_model.pth'

# 训练参数
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# MNIST数据集的类别名称
CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 
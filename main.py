"""
CNN实验项目主入口
包含命令行接口，可以进行训练、测试和可视化
"""
import os
import argparse
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from config import MODEL_CONFIGS, MLP_CONFIGS, EPOCHS, SAVE_PATH, MNIST_CLASS_NAMES, DDR_CLASS_NAMES
from train import train_single_model, train_compare_models
from test import test_model, visualize_model_features
from models.cnn_model import get_cnn_model
from utils.model_visualization import visualize_model_structure
from utils.visualization import (
    plot_confusion_matrix, plot_learning_curves, plot_model_comparison,
    visualize_feature_maps, visualize_all_conv_layers, visualize_filters,
    visualize_misclassified
)
from utils.data_loader import get_dataloaders


def main():
    """主函数，解析命令行参数并执行相应操作"""
    parser = argparse.ArgumentParser(description='CNN实验项目')
    parser.add_argument('action', type=str, choices=['train', 'test', 'visualize', 'structure'],
                        help='要执行的操作：train(训练)、test(测试)、visualize(可视化)、structure(可视化模型结构)')
    parser.add_argument('--model_type', type=str, default='CNN', choices=['CNN', 'resnet', 'mlp'],
                        help='CNN、resnet(残差网络)、mlp(多层感知机)')
    parser.add_argument('--config', type=str, default='simple', choices=['simple', 'medium', 'complex'],
                        help='模型配置：simple(简单)、medium(中等)或complex(复杂)')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'ddr'],
                        help='数据集：mnist或ddr')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'训练轮次 (默认: {EPOCHS})')
    parser.add_argument('--compare', action='store_true',
                        help='训练时比较所有模型配置')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型文件路径，用于测试或可视化')
    parser.add_argument('--output_path', type=str, default='./figures/model_arch',
                        help='模型结构可视化输出路径')
    parser.add_argument('--use_best', action='store_true',
                        help='使用验证集最佳模型进行测试（默认使用最后一轮模型）')
    parser.add_argument('--visualize_type', type=str, choices=['feature_maps', 'filters', 'all'],
                        help='可视化类型：feature_maps(特征图)、filters(滤波器)、all(所有)')
    
    args = parser.parse_args()
    
    # 确保模型保存路径存在
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # 确保输出路径存在
    os.makedirs(args.output_path, exist_ok=True)
    
    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 根据命令行参数执行相应操作
    if args.action == 'train':
        if args.compare:
            print("\n开始训练并比较多个模型配置...")
            results = train_compare_models(
                dataset=args.dataset,
                epochs=args.epochs
            )
            print("\n模型训练完成！")
        else:
            print(f"\n开始训练 {args.model_type} ({args.config}) 模型...")
            trainer = train_single_model(
                model_type=args.model_type,
                config_name=args.config,
                dataset=args.dataset,
                epochs=args.epochs
            )
            print("\n模型训练完成！")
    
    elif args.action == 'test':
        # 确定模型路径
        if args.model_path:
            model_path = args.model_path
        else:
            model_name = f"{args.model_type}_{args.config}"
            if args.use_best:
                model_path = os.path.join(SAVE_PATH, f"{model_name}_best.pth")
            else:
                model_path = os.path.join(SAVE_PATH, f"{model_name}_last.pth")
        
        print(f"\n开始测试模型: {model_path}")
        
        # 确定模型配置
        if args.model_type == 'CNN':
            config = MODEL_CONFIGS[args.config]
        elif args.model_type == 'resnet':
            if args.config == 'simple':
                config = {'num_blocks': [1, 1, 1]}
            elif args.config == 'medium':
                config = {'num_blocks': [2, 2, 2]}
            else:  # complex
                config = {'num_blocks': [3, 4, 6]}
        elif args.model_type == 'mlp':
            config = MLP_CONFIGS[args.config]
        else:
            raise ValueError(f"不支持的模型类型: {args.model_type}")
        
        # 确定类别名称
        class_names = DDR_CLASS_NAMES if args.dataset == 'ddr' else MNIST_CLASS_NAMES
        
        # 获取数据加载器
        train_loader, val_loader, test_loader = get_dataloaders(args.dataset)
        
        # 测试模型
        test_loss, test_acc, all_preds, all_labels, test_losses, test_accs = test_model(
            model_path=model_path,
            model_type=args.model_type,
            config=config,
            dataset=args.dataset,
            class_names=class_names
        )
        
        # 打印测试结果
        print(f"\n测试结果:")
        print(f"测试损失: {test_loss:.4f}")
        print(f"测试准确率: {test_acc:.4f}")
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        # 绘制测试曲线
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(test_losses, label='测试损失')
        plt.xlabel('批次')
        plt.ylabel('损失')
        plt.title('测试损失曲线')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(test_accs, label='测试准确率')
        plt.xlabel('批次')
        plt.ylabel('准确率')
        plt.title('测试准确率曲线')
        plt.legend()
        
        plt.tight_layout()
        save_path = os.path.join(args.output_path, f"{args.model_type}_{args.config}_test_curves.png")
        plt.savefig(save_path)
        print(f"\n测试曲线图已保存到: {save_path}")
        plt.close()
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        plot_confusion_matrix(cm, class_names)
        plt.title(f"{args.model_type.capitalize()} ({args.config}) 模型混淆矩阵")
        plt.tight_layout()
        
        # 保存混淆矩阵图
        save_path = os.path.join(args.output_path, f"{args.model_type}_{args.config}_confusion_matrix.png")
        plt.savefig(save_path)
        print(f"\n混淆矩阵图已保存到: {save_path}")
        plt.close()
        
        # 可视化错误分类样本
        print("\n可视化错误分类样本...")
        save_path = os.path.join(args.output_path, f"{args.model_type}_{args.config}_misclassified.png")
        
        # 创建模型实例
        if args.model_type == 'CNN':
            model = get_cnn_model(
                model_type=args.model_type,
                num_filters=config['num_filters'],
                num_layers=config['num_layers']
            )
        elif args.model_type == 'resnet':
            model = get_cnn_model(
                model_type=args.model_type,
                num_blocks=config['num_blocks']
            )
        elif args.model_type == 'mlp':
            model = get_cnn_model(
                model_type=args.model_type,
                hidden_sizes=config['hidden_sizes'],
                dropout_rate=config['dropout_rate']
            )
        
        # 加载模型状态字典
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        visualize_misclassified(model, test_loader, device, save_path=save_path)
    
    elif args.action == 'visualize':
        # 确定模型路径
        if args.model_path:
            model_path = args.model_path
        else:
            model_name = f"{args.model_type}_{args.config}"
            if args.use_best:
                model_path = os.path.join(SAVE_PATH, f"{model_name}_best.pth")
            else:
                model_path = os.path.join(SAVE_PATH, f"{model_name}_last.pth")
        
        print(f"\n开始可视化模型: {model_path}")
        
        # 确定模型配置
        if args.model_type == 'CNN':
            config = MODEL_CONFIGS[args.config]
        elif args.model_type == 'resnet':
            if args.config == 'simple':
                config = {'num_blocks': [1, 1, 1]}
            elif args.config == 'medium':
                config = {'num_blocks': [2, 2, 2]}
            else:  # complex
                config = {'num_blocks': [3, 4, 6]}
        elif args.model_type == 'mlp':
            config = MLP_CONFIGS[args.config]
        else:
            raise ValueError(f"不支持的模型类型: {args.model_type}")
        
        # 创建模型实例
        if args.model_type == 'CNN':
            model = get_cnn_model(
                model_type=args.model_type,
                num_filters=config['num_filters'],
                num_layers=config['num_layers']
            )
        elif args.model_type == 'resnet':
            model = get_cnn_model(
                model_type=args.model_type,
                num_blocks=config['num_blocks']
            )
        elif args.model_type == 'mlp':
            model = get_cnn_model(
                model_type=args.model_type,
                hidden_sizes=config['hidden_sizes'],
                dropout_rate=config['dropout_rate']
            )
        
        # 加载模型状态字典
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # 获取数据加载器
        train_loader, val_loader, test_loader = get_dataloaders(args.dataset)
        
        # 获取一个样本用于可视化
        sample_image, _ = next(iter(test_loader))
        sample_image = sample_image[0].to(device)
        
        # 根据可视化类型执行相应的可视化
        if args.visualize_type == 'feature_maps' or args.visualize_type == 'all':
            print("\n可视化特征图...")
            save_path = os.path.join(args.output_path, f"{args.model_type}_{args.config}_feature_maps.png")
            visualize_feature_maps(model, sample_image, device, save_path=save_path)
        
        if args.visualize_type == 'filters' or args.visualize_type == 'all':
            print("\n可视化滤波器...")
            save_path = os.path.join(args.output_path, f"{args.model_type}_{args.config}_filters.png")
            visualize_filters(model, save_path=save_path)
        
        if args.visualize_type == 'all':
            print("\n可视化所有卷积层...")
            save_dir = os.path.join(args.output_path, f"{args.model_type}_{args.config}_conv_layers")
            os.makedirs(save_dir, exist_ok=True)
            visualize_all_conv_layers(model, sample_image, device, save_dir=save_dir)
    
    elif args.action == 'structure':
        # 确定模型配置
        if args.model_type == 'CNN':
            config = MODEL_CONFIGS[args.config]
        elif args.model_type == 'resnet':
            if args.config == 'simple':
                config = {'num_blocks': [1, 1, 1]}
            elif args.config == 'medium':
                config = {'num_blocks': [2, 2, 2]}
            else:  # complex
                config = {'num_blocks': [3, 4, 6]}
        elif args.model_type == 'mlp':
            config = MLP_CONFIGS[args.config]
        else:
            if args.config == 'simple':
                config = {'num_blocks': [1, 1, 1]}
            elif args.config == 'medium':
                config = {'num_blocks': [2, 2, 2]}
            else:  # complex
                config = {'num_blocks': [3, 4, 6]}
        
        # 可视化模型结构
        visualize_model_structure(
            model_type=args.model_type,
            config=config,
            output_path=args.output_path
        )


if __name__ == "__main__":
    main() 
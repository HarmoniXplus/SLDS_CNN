"""
模型可视化模块
使用PlotNeuralNet生成神经网络结构图
"""
import os
import sys
import torch
import subprocess
from pathlib import Path
from models.cnn_model import get_cnn_model
from config import MODEL_CONFIGS, MLP_CONFIGS

# 添加PlotNeuralNet目录到系统路径
plot_neural_net_path = Path('./PlotNeuralNet')
sys.path.append(str(plot_neural_net_path))

# 检查目录是否存在
if not plot_neural_net_path.exists():
    raise ImportError("未找到PlotNeuralNet目录，请先运行: git clone https://github.com/HarisIqbal88/PlotNeuralNet.git")

# 导入需要的PlotNeuralNet模块
try:
    from PlotNeuralNet.pycore.tikzeng import *
    from PlotNeuralNet.pycore.blocks import *
except ImportError as e:
    print(f"无法导入PlotNeuralNet模块，错误: {e}")
    print("请确保已正确安装 PlotNeuralNet")
    sys.exit(1)

# 新增torchview可视化
try:
    from torchview import draw_graph
except ImportError:
    raise ImportError("请先安装torchview库: pip install torchview")

# 定义可能缺失的函数
def to_FC(name, n_output, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, caption=" "):
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption="""+ caption +r""",
        fill=\ShadeColor!10,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
\draw [connection]  ("""+name+r"""-west)-- node {\small """+ str(n_output) +"""} ("""+name+r"""-east);
"""

def to_BatchNorm(name, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=25, depth=25, opacity=0.8):
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {RightBandedBox={
        name=""" + name +r""",
        caption=BN,
        xlabel={{" ","dummy"}},
        zlabel="""+ str() +r""",
        fill=\BatchNormColor,
        bandfill=\BatchNormColor,
        opacity="""+ str(opacity) +r""",
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_Sum(name, offset="(0,0,0)", to="(0,0,0)", radius=2.5, opacity=0.6):
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {Ball={
        name=""" + name +r""",
        caption=+,
        fill=\SumColor,
        opacity="""+ str(opacity) +r""",
        radius="""+ str(radius) +r""",
        logo=$+$
        }
    };
"""

def to_ReLU(name, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=25, depth=25, opacity=0.8):
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {RightBandedBox={
        name=""" + name +r""",
        caption=ReLU,
        xlabel={{" ","dummy"}},
        zlabel="""+ str() +r""",
        fill=\ReluColor,
        bandfill=\ReluColor,
        opacity="""+ str(opacity) +r""",
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_Conv(name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=2, height=40, depth=40, caption=" "):
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption="""+ caption +r""",
        xlabel={{"""+ str(n_filer) +""", }},
        zlabel="""+ str(s_filer) +r""",
        fill=\ConvColor,
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_Pool(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    return r"""
\pic[shift={"""+ offset +r"""}] at """+ to +r""" 
    {Box={
        name=""" + name +r""",
        caption="""+ caption +r""",
        fill=\PoolColor,
        opacity="""+ str(opacity) +r""",
        height="""+ str(height) +r""",
        width="""+ str(width) +r""",
        depth="""+ str(depth) +r"""
        }
    };
"""

def to_cor():
    return r"""
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\FcReluColor{rgb:blue,5;red,5;white,4}
\def\SoftmaxColor{rgb:magenta,5;black,7}   
\def\SumColor{rgb:blue,5;green,15}
\def\BatchNormColor{rgb:green,5;blue,2;black,0.3}
\def\ReluColor{rgb:red,5;blue,1;black,0.3}
\def\ShadeColor{rgb:blue,5;red,2.5;white,5}
"""

def check_latex_installed():
    """检查是否安装了LaTeX"""
    try:
        subprocess.run(['pdflatex', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def visualize_simple_cnn(num_layers=2, num_filters=32, output_path='./figures/model_arch'):
    """
    可视化SimpleCNN模型结构
    
    参数:
        num_layers: 卷积层数量
        num_filters: 初始卷积层滤波器数量
        output_path: 输出文件路径（不含扩展名）
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建架构文件
    arch = [
        to_head('../PlotNeuralNet'),
        to_cor(),
        to_begin()
    ]
    
    # 输入层
    arch.append(to_input('mnist.jpg'))
    
    # 动态添加卷积层、批归一化层、激活层和池化层
    in_channel = 1  # MNIST为单通道灰度图
    h_size = 28  # MNIST图像尺寸
    w_size = 28
    
    # 跟踪前一层的名称
    prev_layer = "0"  # 输入层的名称
    
    for i in range(num_layers):
        # 卷积层
        conv_name = f"conv{i+1}"
        arch.append(to_Conv(conv_name, s_filer=3, n_filer=num_filters*(2**i), 
                           offset="(0,0,0)", to=f"({prev_layer})", width=1, height=40, depth=40))
        arch.append(to_connection(prev_layer, conv_name))
        
        # 批归一化层
        bn_name = f"bn{i+1}"
        arch.append(to_BatchNorm(bn_name, offset="(0,0,0)", to=f"({conv_name})", 
                               width=1, height=40, depth=40))
        arch.append(to_connection(conv_name, bn_name))
        
        # ReLU激活层
        relu_name = f"relu{i+1}"
        arch.append(to_ReLU(relu_name, offset="(0,0,0)", to=f"({bn_name})", 
                          width=1, height=40, depth=40))
        arch.append(to_connection(bn_name, relu_name))
        
        # 最大池化层
        h_size = h_size // 2
        w_size = w_size // 2
        pool_name = f"pool{i+1}"
        arch.append(to_Pool(pool_name, offset="(0,0,0)", to=f"({relu_name})", 
                          width=1, height=h_size, depth=w_size))
        arch.append(to_connection(relu_name, pool_name))
        
        # 更新前一层名称
        prev_layer = pool_name
        
        # 更新通道数量
        in_channel = num_filters * (2**i)
    
    # 全连接层
    feature_size = 28 // (2 ** num_layers)
    fc_input_size = in_channel * feature_size * feature_size
    
    # 展平层
    flatten_name = "flatten"
    arch.append(to_SoftMax(flatten_name, 784, "(0,0,0)", f"({prev_layer})", 
                          width=1, height=4, depth=4, caption=""))
    arch.append(to_connection(prev_layer, flatten_name))
    
    # FC1层
    fc1_name = "fc1"
    arch.append(to_FC(fc1_name, 128, "(0,0,0)", f"({flatten_name})", width=1, height=4, depth=4))
    arch.append(to_connection(flatten_name, fc1_name))
    
    # Dropout层
    dropout_name = "dropout"
    arch.append(to_SoftMax(dropout_name, 128, "(0,0,0)", f"({fc1_name})", 
                          width=1, height=4, depth=4, caption="Dropout 0.5"))
    arch.append(to_connection(fc1_name, dropout_name))
    
    # FC2层
    fc2_name = "fc2"
    arch.append(to_FC(fc2_name, 10, "(0,0,0)", f"({dropout_name})", width=1, height=4, depth=4))
    arch.append(to_connection(dropout_name, fc2_name))
    
    arch.append(to_end())
    
    # 写入文件
    tikz_file = output_path + ".tex"
    with open(tikz_file, 'w') as f:
        for item in arch:
            f.write(item)
    
    # 如果安装了LaTeX，则编译生成PDF
    if check_latex_installed():
        # 切换到文件所在目录
        original_dir = os.getcwd()
        os.chdir(os.path.dirname(output_path))
        
        try:
            # 编译LaTeX文件
            result = subprocess.run(['pdflatex', '-interaction=nonstopmode', 
                                     os.path.basename(tikz_file)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"编译结果: {'成功' if result.returncode == 0 else '失败'}")
            
            # 即使编译失败，也尝试生成PDF文件
            pdf_file = os.path.basename(output_path) + ".pdf"
            
            # 检查PDF文件是否存在
            if os.path.exists(pdf_file):
                print(f"模型结构图已生成: {os.path.join(os.path.dirname(output_path), pdf_file)}")
                
                # 尝试将PDF转换为PNG
            png_file = os.path.basename(output_path) + ".png"
            try:
                # 尝试将PDF转换为PNG (需要安装ImageMagick)
                subprocess.run(['convert', pdf_file, png_file], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"生成PNG文件: {os.path.join(os.path.dirname(output_path), png_file)}")
            except Exception as e:
                print(f"无法将PDF转换为PNG: {e}")
                print("请手动转换或安装ImageMagick")
            else:
                print(f"PDF文件未生成，请检查LaTeX错误")
        
        except Exception as e:
            print(f"编译过程中出错: {e}")
        
        finally:
            # 恢复原始工作目录
            os.chdir(original_dir)
        
        return output_path + ".pdf"
    else:
        print("未检测到LaTeX安装，已生成.tex文件，请手动编译")
        print(f"生成的.tex文件: {tikz_file}")
        return tikz_file


def visualize_mlp(hidden_sizes=[128, 64], dropout_rate=0.5, output_path='./figures/mlp_arch'):
    """
    可视化MLP模型结构
    
    参数:
        hidden_sizes: 隐藏层大小列表
        dropout_rate: Dropout比率
        output_path: 输出文件路径（不含扩展名）
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建架构文件
    arch = [
        to_head('../PlotNeuralNet'),
        to_cor(),
        to_begin()
    ]
    
    # 输入层
    arch.append(to_input('mnist.jpg'))
    
    # 展平层
    arch.append(to_SoftMax("flatten", 784, "(0,0,0)", "(0,0,0)", 
                          width=1, height=4, depth=4, caption="Flatten"))
    arch.append(to_connection("0", "flatten"))
    
    # 动态添加全连接层和Dropout层
    prev_layer = "flatten"
    prev_size = 784  # 展平后的MNIST图像大小
    
    for i, hidden_size in enumerate(hidden_sizes):
        # 全连接层
        fc_name = f"fc{i+1}"
        arch.append(to_FC(fc_name, hidden_size, "(0,0,0)", f"({prev_layer})", 
                         width=1, height=4, depth=4))
        arch.append(to_connection(prev_layer, fc_name))
        
        # ReLU激活层
        relu_name = f"relu{i+1}"
        arch.append(to_ReLU(relu_name, "(0,0,0)", f"({fc_name})", 
                          width=1, height=4, depth=4))
        arch.append(to_connection(fc_name, relu_name))
        
        # Dropout层
        dropout_name = f"dropout{i+1}"
        arch.append(to_SoftMax(dropout_name, hidden_size, "(0,0,0)", f"({relu_name})", 
                              width=1, height=4, depth=4, 
                              caption=f"Dropout {dropout_rate}"))
        arch.append(to_connection(relu_name, dropout_name))
        
        # 更新前一层
        prev_layer = dropout_name
        prev_size = hidden_size
    
    # 输出层
    arch.append(to_FC("output", 10, "(0,0,0)", f"({prev_layer})", 
                    width=1, height=4, depth=4))
    arch.append(to_connection(prev_layer, "output"))
    
    arch.append(to_end())
    
    # 写入文件
    tikz_file = output_path + ".tex"
    with open(tikz_file, 'w') as f:
        for item in arch:
            f.write(item)
    
    # 如果安装了LaTeX，则编译生成PDF
    if check_latex_installed():
        # 切换到文件所在目录
        original_dir = os.getcwd()
        os.chdir(os.path.dirname(output_path))
        
        try:
            # 编译LaTeX文件
            result = subprocess.run(['pdflatex', '-interaction=nonstopmode', 
                                     os.path.basename(tikz_file)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"编译结果: {'成功' if result.returncode == 0 else '失败'}")
            
            # 即使编译失败，也尝试生成PDF文件
            pdf_file = os.path.basename(output_path) + ".pdf"
            
            # 检查PDF文件是否存在
            if os.path.exists(pdf_file):
                print(f"模型结构图已生成: {os.path.join(os.path.dirname(output_path), pdf_file)}")
                
                # 尝试将PDF转换为PNG
            png_file = os.path.basename(output_path) + ".png"
            try:
                # 尝试将PDF转换为PNG (需要安装ImageMagick)
                subprocess.run(['convert', pdf_file, png_file], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                print(f"无法将PDF转换为PNG: {e}")
                print("请手动转换或安装ImageMagick")
            else:
                print(f"PDF文件未生成，请检查LaTeX错误")
        
        except Exception as e:
            print(f"编译过程中出错: {e}")
        
        finally:
            # 恢复原始工作目录
            os.chdir(original_dir)
        
        return output_path + ".pdf"
    else:
        print("未检测到LaTeX安装，已生成.tex文件，请手动编译")
        print(f"生成的.tex文件: {tikz_file}")
        return tikz_file


def visualize_resnet(num_blocks=[2, 2, 2], output_path='./figures/resnet_arch'):
    """
    可视化ResNet模型结构
    
    参数:
        num_blocks: 每个阶段的残差块数量
        output_path: 输出文件路径（不含扩展名）
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建架构文件
    arch = [
        to_head('../PlotNeuralNet'),
        to_cor(),
        to_begin()
    ]
    
    # 输入层
    arch.append(to_input('mnist.jpg'))
    
    # 第一个卷积层
    arch.append(to_Conv("conv1", 3, 16, "(0,0,0)", "(0,0,0)", 1, 40, 40))
    arch.append(to_connection("0", "conv1"))
    
    # 批归一化和ReLU
    arch.append(to_BatchNorm("bn1", "(0,0,0)", "(conv1)", 1, 40, 40))
    arch.append(to_connection("conv1", "bn1"))
    
    arch.append(to_ReLU("relu1", "(0,0,0)", "(bn1)", 1, 40, 40))
    arch.append(to_connection("bn1", "relu1"))
    
    # 当前层名称
    prev_layer = "relu1"
    in_channels = 16
    
    # 动态添加残差块
    for i, block_count in enumerate(num_blocks):
        out_channels = 16 * (2 ** i)
        
        for j in range(block_count):
            stride = 2 if (j == 0 and i > 0) else 1
            
            # 创建残差块
            block_name = f"resblock_{i+1}_{j+1}"
            
            # 如果需要下采样，创建一个分支
            if stride != 1 or in_channels != out_channels:
                shortcut_name = f"{block_name}_shortcut"
                arch.append(to_Conv(shortcut_name, 1, out_channels, 
                                  "(0,0,0)", f"({prev_layer})", 1, 40, 40))
                arch.append(to_connection(prev_layer, shortcut_name))
                
                shortcut_bn_name = f"{shortcut_name}_bn"
                arch.append(to_BatchNorm(shortcut_bn_name, 
                                       "(0,0,0)", f"({shortcut_name})", 1, 40, 40))
                arch.append(to_connection(shortcut_name, shortcut_bn_name))
                
                shortcut_end = shortcut_bn_name
            else:
                shortcut_end = prev_layer
            
            # 残差块的主路径
            conv1_name = f"{block_name}_conv1"
            arch.append(to_Conv(conv1_name, 3, out_channels, 
                              "(0,0,0)", f"({prev_layer})", 1, 40, 40))
            arch.append(to_connection(prev_layer, conv1_name))
            
            bn1_name = f"{block_name}_bn1"
            arch.append(to_BatchNorm(bn1_name, 
                                   "(0,0,0)", f"({conv1_name})", 1, 40, 40))
            arch.append(to_connection(conv1_name, bn1_name))
            
            relu1_name = f"{block_name}_relu1"
            arch.append(to_ReLU(relu1_name, 
                              "(0,0,0)", f"({bn1_name})", 1, 40, 40))
            arch.append(to_connection(bn1_name, relu1_name))
            
            conv2_name = f"{block_name}_conv2"
            arch.append(to_Conv(conv2_name, 3, out_channels, 
                              "(0,0,0)", f"({relu1_name})", 1, 40, 40))
            arch.append(to_connection(relu1_name, conv2_name))
            
            bn2_name = f"{block_name}_bn2"
            arch.append(to_BatchNorm(bn2_name, 
                                   "(0,0,0)", f"({conv2_name})", 1, 40, 40))
            arch.append(to_connection(conv2_name, bn2_name))
            
            # 添加残差连接
            block_end = f"{block_name}_end"
            arch.append(to_Sum(block_end, 
                             "(0,0,0)", f"({bn2_name})"))
            arch.append(to_connection(bn2_name, block_end))
            arch.append(to_connection(shortcut_end, block_end))
            
            # 最后的ReLU
            block_relu = f"{block_name}_relu2"
            arch.append(to_ReLU(block_relu, 
                              "(0,0,0)", f"({block_end})", 1, 40, 40))
            arch.append(to_connection(block_end, block_relu))
            
            # 更新前一层名称和通道数
            prev_layer = block_relu
            in_channels = out_channels
    
    # 全局平均池化
    arch.append(to_Pool("avg_pool", "(0,0,0)", f"({prev_layer})", 1, 4, 4))
    arch.append(to_connection(prev_layer, "avg_pool"))
    
    # 全连接分类层
    arch.append(to_FC("fc", 10, "(0,0,0)", "(avg_pool)", 1, 4, 4))
    arch.append(to_connection("avg_pool", "fc"))
    
    arch.append(to_end())
    
    # 写入文件
    tikz_file = output_path + ".tex"
    with open(tikz_file, 'w') as f:
        for item in arch:
            f.write(item)
    
    # 如果安装了LaTeX，则编译生成PDF
    if check_latex_installed():
        # 切换到文件所在目录
        original_dir = os.getcwd()
        os.chdir(os.path.dirname(output_path))
        
        try:
            # 编译LaTeX文件
            result = subprocess.run(['pdflatex', '-interaction=nonstopmode', 
                                     os.path.basename(tikz_file)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"编译结果: {'成功' if result.returncode == 0 else '失败'}")
            
            # 即使编译失败，也尝试生成PDF文件
            pdf_file = os.path.basename(output_path) + ".pdf"
            
            # 检查PDF文件是否存在
            if os.path.exists(pdf_file):
                print(f"模型结构图已生成: {os.path.join(os.path.dirname(output_path), pdf_file)}")
                
                # 尝试将PDF转换为PNG
            png_file = os.path.basename(output_path) + ".png"
            try:
                # 尝试将PDF转换为PNG (需要安装ImageMagick)
                subprocess.run(['convert', pdf_file, png_file], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                print(f"无法将PDF转换为PNG: {e}")
                print("请手动转换或安装ImageMagick")
            else:
                print(f"PDF文件未生成，请检查LaTeX错误")
        
        except Exception as e:
            print(f"编译过程中出错: {e}")
        
        finally:
            # 恢复原始工作目录
            os.chdir(original_dir)
        
        return output_path + ".pdf"
    else:
        print("未检测到LaTeX安装，已生成.tex文件，请手动编译")
        print(f"生成的.tex文件: {tikz_file}")
        return tikz_file


def visualize_model_structure(model_type='CNN', config=None, output_path='./figures/model_arch'):
    """
    使用torchview可视化模型结构，支持CNN/resnet/mlp
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if model_type == 'CNN':
        if config is None:
            config = MODEL_CONFIGS['simple']
        model = get_cnn_model(model_type='CNN', num_filters=config['num_filters'], num_layers=config['num_layers'])
        dummy = torch.randn(1, 1, 28, 28)
    elif model_type == 'resnet':
        if config is None:
            config = {'num_blocks': [2, 2, 2]}
        model = get_cnn_model(model_type='resnet', num_blocks=config['num_blocks'])
        dummy = torch.randn(1, 1, 28, 28)
    elif model_type == 'mlp':
        from models.cnn_model import MLP
        if config is None:
            config = MLP_CONFIGS['simple']
        model = MLP(num_classes=10, hidden_sizes=config['hidden_sizes'], dropout_rate=config['dropout_rate'])
        dummy = torch.randn(1, 1, 28, 28)
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    # torchview生成结构图
    graph = draw_graph(
        model,
        input_data=dummy,
        expand_nested=True,
        save_graph=True,
        filename=output_path,
        directory=os.path.dirname(output_path),
        graph_name=f"{model_type}_structure"
    )
    print(f"模型结构图已保存到 {output_path}.png")
    return output_path + ".png" 
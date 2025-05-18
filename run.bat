@echo off
echo CNN手写数字识别实验项目
echo =======================

set /p action="选择操作(1:训练模型 2:测试模型 3:可视化特征 4:可视化模型结构 5:执行所有操作): "

if "%action%"=="1" (
    set /p model_type="选择模型类型(simple/resnet/mlp): "
    set /p config="选择模型配置(simple/medium/complex): "
    set /p epochs="训练轮次(默认20): "
    set /p compare="是否比较所有模型(y/n): "
    
    if "%compare%"=="y" (
        python main.py train --compare
    ) else (
        if "%epochs%"=="" set epochs=20
        python main.py train --model_type %model_type% --config %config% --epochs %epochs%
    )
) else if "%action%"=="2" (
    set /p model_type="选择模型类型(simple/resnet/mlp): "
    set /p config="选择模型配置(simple/medium/complex): "
    python main.py test --model_type %model_type% --config %config%
) else if "%action%"=="3" (
    set /p model_type="选择模型类型(simple/resnet/mlp): "
    set /p config="选择模型配置(simple/medium/complex): "
    python main.py visualize --model_type %model_type% --config %config%
) else if "%action%"=="4" (
    set /p model_type="选择模型类型(simple/resnet/mlp): "
    set /p config="选择模型配置(simple/medium/complex): "
    set /p output_path="输出路径(默认./figures/model_arch): "
    
    if "%output_path%"=="" set output_path=./figures/model_arch
    python main.py structure --model_type %model_type% --config %config% --output_path %output_path%
) else if "%action%"=="5" (
    set /p model_type="选择模型类型(simple/resnet/mlp): "
    set /p config="选择模型配置(simple/medium/complex): "
    set /p epochs="训练轮次(默认20): "
    
    if "%epochs%"=="" set epochs=20
    python main.py all --model_type %model_type% --config %config% --epochs %epochs%
) else (
    echo 无效选择，请选择 1-5
)

pause 
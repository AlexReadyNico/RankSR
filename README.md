# RankSR

可用的骨干网络包括：

- [x] VGGNet-13/16/19
- [x] EfficientNet系列；
- [x] ResNet系列;

```

### 模型训练


```shell
python main.py --data data/ --train --arch efficientnet_b0 --num_classes 1 \
--criterion=rank --margin 0.1 -- --use_margin --image_size 244 224 \
--pretrained --warmup 5 --epochs 65 -b 384 -j 16 --gpus 1 --nodes 1
```
----------------------------------Rank阶段训练----------------------------------------------
```shell
python main.py --data data/ --train --arch efficientnet_b0 --num_classes 1 \
--criterion=rank --margin 0.1 --image_size 224 224 \
--warmup 5 --epochs 65 -b 1 -j 16 --gpus 1 --nodes 1
```

```shell
python mainEfficientB4.py --data data/ --train --arch efficientnet_b4 --num_classes 1 \
--criterion=rank --margin 1.0 --image_size 540 540 \
--warmup 5 --epochs 5 -b 4 -j 16 --gpus 1 --nodes 1 
```


```shell
python mainResNet.py --data data/ --train --arch ResNet18 --num_classes 1 \
--criterion=rank --margin 1.0 --image_size 540 540 \
--warmup 5 --epochs 1 -b 20 -j 16 --gpus 1 --nodes 1 --resume checkpoints/ResNet18_Margin1.0_Epoch4/checkpoint_ResNet18.pth
```


```shell
python mainVggNet.py --data data/ --train --arch Vgg16 --num_classes 1 \
--criterion=rank --margin 1.0 --image_size 540 540 \
--warmup 5 --epochs 1 -b 20 -j 16 --gpus 1 --nodes 1 --resume checkpoints/VGG16_Margin1.0_Epoch4/checkpoint_Vgg16.pth
```


```shell
python main.py --data data/ --train --arch efficientnet_b0 --num_classes 1 \
--criterion=rank --margin 10 --image_size 224 224 \
--warmup 5 --epochs 65 -b 1 -j 16 --gpus 1 --nodes 1 --resume checkpoints/rank_sr_540_20210624/model_best_efficientnet_b0.pth
```

----------------------------------Regress阶段训练----------------------------------------------
```shell
python main.py --data data/ --train --arch efficientnet_b0 --num_classes 1 \
--criterion=regress --margin 0.1 --image_size 244 224 \
--warmup 5 --epochs 65 -b 1 -j 16 --gpus 1 --nodes 1 --resume checkpoints/Rank/model_best_efficientnet_b0.pth
```
```shell
python main.py --data data/ --train --arch efficientnet_b2 --num_classes 1 \
--criterion=regress --margin 1.0 --image_size 540 540 \
--warmup 5 --epochs 200 -b 20 -j 16 --gpus 1 --nodes 1 --resume checkpoints/EfficientB2_Margin1.0_Epoch6/model_best_efficientnet_b2.pth
```
```shell
python mainResNet.py --data data/ --train --arch ResNet18 --num_classes 1 \
--criterion=regress --margin 1.0 --image_size 540 540 \
--warmup 5 --epochs 100 -b 20 -j 16 --gpus 1 --nodes 1 --resume checkpoints/ResNet18_regress_Epoch100/checkpoint_ResNet18.pth
```

参数的详细说明可查看`config.py`文件。

### 模型评估

基于`data/`目录下的`test`数据集，评估`checkpoints/model_best_efficientnet-b0.pth`目录下的模型
（需要指定模型输入、输出、损失函数、模型结构，数据加载的worker，推理时的batch size）：

----------------------------------Rank阶段Evaluate----------------------------------------------
```shell
python main.py --data data -e --arch efficientnet_b0 --num_classes 1 --criterion=rank --margin 0.1 \
--image_size 224 224 --batch_size 1 --workers 0 --resume checkpoints/model_best_efficientnet_b0.pth -g 1 -n 1
```
```shell
python main.py --data data -e --arch efficientnet_b0 --num_classes 1 --criterion=rank --margin 10 \
--image_size 540 540 --batch_size 12 --workers 0 --resume checkpoints/SR4K_540_Epoch65/checkpoint_efficientnet_b0.pth -g 1 -n 1
```
```shell
python main.py --data data -e --arch efficientnet_b0 --num_classes 1 --criterion=rank --margin 10 \
--image_size 224 224 --batch_size 32 --workers 0 --resume checkpoints/model_best_efficientnet_b0.pth -g 1 -n 1
```
----------------------------------Regress阶段Evaluate----------------------------------------------
```shell
python main.py --data data -e --arch efficientnet_b0 --num_classes 1 --criterion=regress --margin 10 \
--image_size 540 540 --batch_size 16 --workers 0 --resume checkpoints/model_best_efficientnet_b0.pth -g 1 -n 1
```
```shell
python main.py --data data -e --arch efficientnet_b0 --num_classes 1 --criterion=regress --margin 0.1 \
--image_size 224 224 --batch_size 1 --workers 0 --resume checkpoints/checkpoint_efficientnet_b0.pth -g 1 -n 1
```
参数的详细说明可查看`config.py`文件。






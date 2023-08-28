# 可控旋律生成
基于论文Shuqi Dai et al. Controllable Deep Melody Generation Via Hierarchical Music Structure Representation的从零复现

## 环境安装
使用conda环境
```
conda create -n py38 python=3.8
conda activate py38
pip install -r requirements.txt
```

## 文件介绍
POP909: 经信息提取处理后的POP909数据集，存储格式为文本文档。可从该地址下载经预处理的POP909数据集：https://github.com/Dsqvival/hierarchical-structure-analysis

MusicFrameworks.py: 按照论文描述方式定义的数据预处理函数包

dataloader.py: 数据集构建程序

model.py: 定义Transformer-LSTM模型

train.py: basic melody生成模型训练

train_rhythm.py: rhythm生成模型训练

train_melody.py: melody生成模型训练

generate.py: 长序列旋律生成

generate_seg: 片段旋律生成

## 模型训练
分别运行train.py train_rhythm.py train_melody.py完成3个生成模型的训练
```
python train.py
python train_rhythm.py
python train_melody.py
```

## 旋律生成
运行generate.py生成长序列旋律，运行generate_seg.py生成旋律片段
```
python generate.py
python generate_seg.py
```

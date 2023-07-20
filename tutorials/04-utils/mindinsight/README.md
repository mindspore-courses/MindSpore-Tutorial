# MindInsight in MindSpore

MindSpore Insight是一款可视化调试调优工具，帮助用户获得更优的模型精度和性能。

通过MindSpore Insight，可以可视化地查看训练过程、优化模型性能、调试精度问题。用户还可以通过MindSpore Insight提供的命令行方便地搜索超参，迁移模型。[官方文档](https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.0/index.html)

<br>

## Usage 

#### 1. 安装MindInsight

#### 2. 训练模型
```bash
$ python main.py
```

#### 3. 打开MindInsight
运行以下命令，然后打开 http://localhost:6006/
```bash
$ mindinsight --logdir='./logs' --port=6006
```

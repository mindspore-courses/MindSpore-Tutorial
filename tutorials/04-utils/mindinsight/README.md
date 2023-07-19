# MindInsight in MindSpore

MindSpore Insight是一款可视化调试调优工具，帮助用户获得更优的模型精度和性能。

通过MindSpore Insight，可以可视化地查看训练过程、优化模型性能、调试精度问题。用户还可以通过MindSpore Insight提供的命令行方便地搜索超参，迁移模型。[官方文档](https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.0/index.html)

<br>

## Usage 

#### 1. Install the dependencies
```bash
$ pip install -r requirements.txt
```

#### 2. Train the model
```bash
$ python main.py
```

#### 3. Open the MindInsight
To run the TensorBoard, open a new terminal and run the command below. Then, open http://localhost:6006/ on your web browser.
```bash
$ mindinsight --logdir='./logs' --port=6006
```

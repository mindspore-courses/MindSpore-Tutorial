import os
import tarfile
import urllib.request

import mindcv
import mindspore
import mindspore.nn as nn
import numpy as np
import mindspore.dataset.transforms as transforms
from mindspore import ops

# ================================================================== #
#                               内容表                                #
# ================================================================== #

# 1. 自动微分例1                            (Line 29 to 55)
# 2. 自动微分例2                            (Line 61 to 100)
# 3. 从numpy中加载数据                      (Line 106 to 113)
# 4. 输入                                  (Line 119 to 148)
# 5. 自定义数据集                           (Line 154 to 158)
# 6. 预训练模型                             (Line 164 to 177)
# 7. 保存和加载模型                          (Line 183 to 189)


# ================================================================== #
#                           1. 自动微分例 1                           #
# ================================================================== #

# 创建Tensor
x = mindspore.Tensor(1.)
b = mindspore.Tensor(3.)
w = mindspore.Tensor(2.)


# 定义网络
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.b = mindspore.Tensor(3.)
        self.w = mindspore.Tensor(2.)

    def construct(self, x, w, b):
        f = w * x + b
        return f


# 计算梯度
grad_op = ops.GradOperation(get_all=True)
net = Net()
grad_fn = grad_op(net, weights=(x, w, b))
grads = grad_fn(x, w, b)
# 打印梯度
print(grads[0])  # x.grad = 2
print(grads[1])  # w.grad = 1
print(grads[2])  # b.grad = 1

# ================================================================== #
#                            2. 自动微分例 2                          #
# ================================================================== #

# 创建形状为(10,3)和(10,2)的随机Tensor.
x = ops.randn(10, 3)
y = ops.randn(10, 2)

# 构造全连接层
linear = nn.Dense(3, 2)
print('w: ', linear.weight)
print('b: ', linear.bias)

# 构造损失函数和优化器
criterion = nn.MSELoss()
optimizer = nn.optim.SGD(linear.trainable_params(), 0.01)


# 定义正向传播函数
def forward(x):
    pred = linear(x)
    loss = criterion(pred, y)
    return pred, loss


# 定义求梯度函数
grad_fn = ops.value_and_grad(forward, None, optimizer.parameters, has_aux=True)

# 正向传播
(pred, loss), grads = grad_fn(x)

# 打印loss.
print('loss: ', loss.asnumpy().item())

# 打印梯度
print('dL/dw: ', grads[0])
print('dL/db: ', grads[1])

optimizer(grads)

# 打印一步梯度下降后的loss
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.asnumpy().item())

# ================================================================== #
#                           3. 从numpy中加载数据                       #
# ================================================================== #

# 创建一个numpy数组
x = np.array([[1, 2], [3, 4]])

# 把numpy数组转化为tensor
y = mindspore.Tensor.from_numpy(x)

# 把tensor转化为numpy数组
z = y.asnumpy()

# ================================================================== #
#                               4. 输入                               #
# ================================================================== #

# 下载导入 CIFAR-10 数据集.
file_path = '../../../data/CIFAR-10'

if not os.path.exists(file_path):
    if not os.path.exists('../../../data'):
        os.mkdir('../../../data')
    # 下载CIFAR-10数据集
    os.mkdir(file_path)
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    file_name = 'cifar-10-binary.tar.gz'
    print("正在从" + url + "下载CIFAR-10数据集...")
    result = urllib.request.urlretrieve(url, os.path.join(file_path, file_name))
    with tarfile.open(os.path.join(file_path, file_name), 'r:gz') as tar:
        print("正在解压数据集...")
        for member in tar.getmembers():
            if member.name.startswith('cifar-10-batches-bin'):
                member.name = os.path.basename(member.name)
                tar.extract(member, path=file_path)
    os.remove(os.path.join(file_path, file_name))

train_dataset = mindspore.dataset.Cifar10Dataset(
    dataset_dir=file_path,
    usage='train',
).map(operations=transforms.vision.ToTensor(), input_columns="image")

# 读取一组数据，创建并使用使用迭代器
for _, (image, label) in enumerate(train_dataset.create_tuple_iterator()):
    print(ops.size(image))
    print(label)
    break

# ================================================================== #
#                            5. 自定义数据集                           #
# ================================================================== #

# 构建自定义数据集使用GeneratorDataset
# 详见https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/dataset/mindspore.dataset.GeneratorDataset.html
dataset = mindspore.dataset.GeneratorDataset(
    # TODO
)

# ================================================================== #
#                             6. 预训练模型                           #
# ================================================================== #

# 从MindCV中加载Resnet18模型
resnet = mindcv.models.resnet18(pretrained=True)

# 如果你只想调整模型的顶层，可以按照以下方式设置
for param in resnet.trainable_params():
    param.requires_grad = False

# 更换顶层进行微调
resnet.classifier = nn.Dense(resnet.classifier.in_channels, 100)

# 前向传播
images = ops.randn(64, 3, 224, 224)
outputs = resnet(images)
print(ops.shape(outputs))  # (64, 100)

# ================================================================== #
#                           7. 保存和加载模型                          #
# ================================================================== #

# Save and load the entire model.
mindspore.save_checkpoint(resnet, 'model.ckpt')
model = mindspore.load_checkpoint('model.ckpt')

# 保存优化器
mindspore.save_checkpoint(optimizer, 'optimizer.ckpt')
state_dict = mindspore.load_checkpoint('optimizer.ckpt')

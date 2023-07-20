"""双向循环神经网络"""
import gzip
import math
import os
import shutil
import urllib.request

import mindspore.common.dtype as mstype
import mindspore.dataset.vision
from mindspore.dataset.vision import transforms
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import HeUniform
import numpy as np


# 设置超参数
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.003

file_path = '../../../data/MNIST/'

if not os.path.exists(file_path):
    if not os.path.exists('../../../data'):
        os.mkdir('../../../data')
    # 下载数据集
    os.mkdir(file_path)
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        url = (base_url + file_name).format(**locals())
        print("正在从" + url + "下载MNIST数据集...")
        urllib.request.urlretrieve(url, os.path.join(file_path, file_name))
        with gzip.open(os.path.join(file_path, file_name), 'rb') as f_in:
            print("正在解压数据集...")
            with open(os.path.join(file_path, file_name)[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(os.path.join(file_path, file_name))

image_transforms = transforms.ToTensor()
label_transforms = transforms.ToTensor(output_type=np.int32)

# 加载MNIST数据集
train_dataset = mindspore.dataset.MnistDataset(
    dataset_dir=file_path,
    usage='train',
    shuffle=True
).map(operations=image_transforms, input_columns="image").batch(batch_size=batch_size)

test_dataset = mindspore.dataset.MnistDataset(
    dataset_dir=file_path,
    usage='test',
    shuffle=False
).map(operations=image_transforms, input_columns="image").batch(batch_size=batch_size)


class BiRNN(nn.Cell):
    """双向循环神经网络"""
    def __init__(self, _input_size, _hidden_size, _num_layers, _num_classes):
        super().__init__()
        self.hidden_size = _hidden_size
        self.num_layers = _num_layers
        # 双向LSTM
        self.lstm = nn.LSTM(_input_size, _hidden_size, _num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Dense(_hidden_size * 2, _num_classes, weight_init=HeUniform(math.sqrt(5)))

    def construct(self, x):
        # 设置初始状态
        hidden0 = ops.zeros(size=(self.num_layers * 2, x.shape[0], self.hidden_size))
        cell0 = ops.zeros(size=(self.num_layers * 2, x.shape[0], self.hidden_size))

        out, _ = self.lstm(x, (hidden0, cell0))

        out = self.linear(out[:, -1, :])
        return out


model = BiRNN(input_size, hidden_size, num_layers, num_classes)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = nn.optim.Adam(model.trainable_params(), learning_rate)

# 绑定损失函数
train_model = nn.WithLossCell(model, loss_fn=criterion)
train_model = nn.TrainOneStepCell(train_model, optimizer)

# 训练
for epoch in range(num_epochs):
    for i, (image, label) in enumerate(train_dataset.create_tuple_iterator()):
        image = ops.reshape(image, (-1, sequence_length, input_size))
        total_step = train_dataset.get_dataset_size()
        train_model.set_train()
        label = mindspore.Tensor(label, mstype.int32)
        loss = train_model(image, label)

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.asnumpy().item():.4f}')

# 测试模型
model.set_train(False)
correct = 0
total = 0
for image, label in test_dataset.create_tuple_iterator():
    image = ops.reshape(image, (-1, sequence_length, input_size))
    label = mindspore.Tensor(label, mstype.int32)
    outputs = model(image)
    _, predicted = ops.max(outputs.value(), 1)
    total += label.shape[0]
    correct += (predicted == label).sum().asnumpy().item()

print(f'Test Accuracy of the model on the 10000 test images: {(100 * correct / total):.2f} %')

# 保存模型
save_path = './rnn.ckpt'
mindspore.save_checkpoint(model, save_path)

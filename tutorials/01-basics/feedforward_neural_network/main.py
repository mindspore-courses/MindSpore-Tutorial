"""前馈神经网络"""
import gzip
import math
import os
import shutil
from urllib import request

import mindspore
import numpy as np
from mindspore import nn, ops
from mindspore.common.initializer import HeUniform
from mindspore.dataset.vision import transforms
import mindspore.common.dtype as mstype

# 超参数
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST数据集
file_path = '../../../data/MNIST/'

if not os.path.exists(file_path):
    # 下载数据集
    if not os.path.exists('../../../data'):
        os.mkdir('../../../data')
    os.mkdir(file_path)
    BASE_URL = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        url = (BASE_URL + file_name).format(**locals())
        print("正在从" + url + "下载MNIST数据集...")
        request.urlretrieve(url, os.path.join(file_path, file_name))
        with gzip.open(os.path.join(file_path, file_name), 'rb') as f_in:
            print("正在解压数据集...")
            with open(os.path.join(file_path, file_name)[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(os.path.join(file_path, file_name))

image_transforms = transforms.ToTensor()
label_transforms = transforms.ToTensor(output_type=np.int32)

train_dataset = mindspore.dataset.MnistDataset(
    dataset_dir=file_path,
    usage='train',
    shuffle=True
).map(operations=image_transforms, input_columns="image").batch(batch_size)

test_dataset = mindspore.dataset.MnistDataset(
    dataset_dir=file_path,
    usage='test',
    shuffle=False
).map(operations=image_transforms, input_columns="image").batch(batch_size=batch_size)


class NeuralNet(nn.Cell):
    """带一个隐藏层的全连接神经网络"""

    def __init__(self, _input_size, _hidden_size, _num_classes):
        super().__init__()
        self.fc1 = nn.Dense(_input_size, _hidden_size, weight_init=HeUniform(math.sqrt(5)))
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(_hidden_size, _num_classes, weight_init=HeUniform(math.sqrt(5)))

    def construct(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = nn.optim.Adam(model.trainable_params(), learning_rate)

# 绑定训练参数
model_with_loss = nn.WithLossCell(model, criterion)
train_model = nn.TrainOneStepCell(model_with_loss, optimizer)

# 训练模型
for epoch in range(num_epochs):
    for i, (image, label) in enumerate(train_dataset.create_tuple_iterator()):
        total_step = train_dataset.get_dataset_size()
        train_model.set_train()
        image = ops.reshape(image, (-1, 28 * 28))
        label = mindspore.Tensor(label, mstype.int32)
        loss = train_model(image, label)
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.asnumpy().item():.4f}')

model.set_train(False)

# 测试模型
correct = 0
total = 0
for image, label in test_dataset.create_tuple_iterator():
    label = mindspore.Tensor(label, mstype.int32)
    image = ops.reshape(image, (-1, 28 * 28))
    outputs = model(image)
    _, predicted = ops.max(outputs.value(), 1)
    total += label.shape[0]
    correct += (predicted == label).sum().asnumpy().item()

print(f'Test Accuracy of the model on the 10000 test images: {(100 * correct / total):.2f} %')

# Save the model checkpoint
save_path = './model.ckpt'
mindspore.save_checkpoint(model, save_path)

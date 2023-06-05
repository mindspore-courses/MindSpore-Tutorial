import gzip
import math
import os
import shutil
import urllib.request

import mindspore.common.dtype as mstype
import mindspore.dataset.vision
import mindspore.dataset.vision.transforms as transforms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore.common.initializer import HeUniform

# 设置超参数
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

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

# RNN网络结构
class RNN(nn.Cell):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Dense(hidden_size, num_classes, weight_init=HeUniform(math.sqrt(5)))

    def construct(self, x):
        # 设置初始状态
        h0 = ops.zeros(size=(self.num_layers, x.shape[0], self.hidden_size))
        c0 = ops.zeros(size=(self.num_layers, x.shape[0], self.hidden_size))

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


model = RNN(input_size, hidden_size, num_layers, num_classes)

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
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.asnumpy().item()))

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

print('Test Accuracy of the model on the 10000 test images: {:.2f} %'.format(100 * correct / total))

# 保存模型
save_path = './birnn.ckpt'
mindspore.save_checkpoint(model, save_path)

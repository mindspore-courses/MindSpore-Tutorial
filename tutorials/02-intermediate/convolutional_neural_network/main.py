"""卷积神经网络"""
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


# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

file_path = '../../../data/MNIST/'

if not os.path.exists(file_path):
    # 下载数据集
    if not os.path.exists('../../../data'):
        os.mkdir('../../../data')
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

train_dataset = mindspore.dataset.MnistDataset(
    dataset_dir=file_path,
    usage='train',
).map(operations=image_transforms, input_columns="image").batch(batch_size)

test_dataset = mindspore.dataset.MnistDataset(
    dataset_dir=file_path,
    usage='test',
).map(operations=image_transforms, input_columns="image").batch(batch_size=batch_size)


class CNN(nn.Cell):
    """卷积神经网络"""
    def __init__(self, _num_classes=10):
        super().__init__()
        self.layer1 = nn.SequentialCell(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.SequentialCell(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.linear = nn.Dense(7 * 7 * 32, _num_classes, weight_init=HeUniform(math.sqrt(5)))

    def construct(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = ops.reshape(out, (ops.shape(out)[0], -1))
        out = self.fc(out)
        return out


model = CNN(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = nn.optim.Adam(model.trainable_params(), learning_rate=learning_rate)
train_model = nn.WithLossCell(model, loss_fn=criterion)
train_model = nn.TrainOneStepCell(train_model, optimizer)

for epoch in range(num_epochs):
    for i, (image, label) in enumerate(train_dataset.create_tuple_iterator()):
        total_step = train_dataset.get_dataset_size()
        train_model.set_train()
        label = mindspore.Tensor(label, mstype.int32)
        loss = train_model(image, label)
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.asnumpy().item():.4f}')

model.set_train(False)

# Test the model
correct = 0
total = 0
for image, label in test_dataset.create_tuple_iterator():
    label = mindspore.Tensor(label, mstype.int32)
    outputs = model(image)
    _, predicted = ops.max(outputs.value(), 1)
    total += label.shape[0]
    correct += (predicted == label).sum().asnumpy().item()

print(f'Test Accuracy of the model on the 10000 test images: {(100 * correct / total):.2f} %')

# Save the model checkpoint
save_path = './cnn.ckpt'
mindspore.save_checkpoint(model, save_path)

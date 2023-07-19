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
NUM_EPOCHS = 5
NUM_CLASSES = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.001

FILE_PATH = '../../../data/MNIST/'

if not os.path.exists(FILE_PATH):
    # 下载数据集
    if not os.path.exists('../../../data'):
        os.mkdir('../../../data')
    os.mkdir(FILE_PATH)
    BASE_URL = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        URL = (BASE_URL + file_name).format(**locals())
        print("正在从" + URL + "下载MNIST数据集...")
        urllib.request.urlretrieve(URL, os.path.join(FILE_PATH, file_name))
        with gzip.open(os.path.join(FILE_PATH, file_name), 'rb') as f_in:
            print("正在解压数据集...")
            with open(os.path.join(FILE_PATH, file_name)[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(os.path.join(FILE_PATH, file_name))

image_transforms = transforms.ToTensor()
label_transforms = transforms.ToTensor(output_type=np.int32)

train_dataset = mindspore.dataset.MnistDataset(
    dataset_dir=FILE_PATH,
    usage='train',
).map(operations=image_transforms, input_columns="image").batch(BATCH_SIZE)

test_dataset = mindspore.dataset.MnistDataset(
    dataset_dir=FILE_PATH,
    usage='test',
).map(operations=image_transforms, input_columns="image").batch(batch_size=BATCH_SIZE)


class CNN(nn.Cell):
    """卷积神经网络"""
    def __init__(self, num_classes=10):
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
        self.linear = nn.Dense(7 * 7 * 32, num_classes, weight_init=HeUniform(math.sqrt(5)))

    def construct(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = ops.reshape(out, (ops.shape(out)[0], -1))
        out = self.fc(out)
        return out


model = CNN(NUM_CLASSES)

criterion = nn.CrossEntropyLoss()
optimizer = nn.optim.Adam(model.trainable_params(), learning_rate=LEARNING_RATE)
train_model = nn.WithLossCell(model, loss_fn=criterion)
train_model = nn.TrainOneStepCell(train_model, optimizer)

for epoch in range(NUM_EPOCHS):
    for i, (image, label) in enumerate(train_dataset.create_tuple_iterator()):
        total_step = train_dataset.get_dataset_size()
        train_model.set_train()
        label = mindspore.Tensor(label, mstype.int32)
        loss = train_model(image, label)
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{total_step}], Loss: {loss.asnumpy().item():.4f}')

model.set_train(False)

# Test the model
CORRECT = 0
TOTAL = 0
for image, label in test_dataset.create_tuple_iterator():
    label = mindspore.Tensor(label, mstype.int32)
    outputs = model(image)
    _, predicted = ops.max(outputs.value(), 1)
    TOTAL += label.shape[0]
    CORRECT += (predicted == label).sum().asnumpy().item()

print(f'Test Accuracy of the model on the 10000 test images: {(100 * CORRECT / TOTAL):.2f} %')

# Save the model checkpoint
SAVE_PATH = './cnn.ckpt'
mindspore.save_checkpoint(model, SAVE_PATH)

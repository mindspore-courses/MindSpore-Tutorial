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
SEQUENCE_LENGTH = 28
INPUT_SIZE = 28
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 10
BATCH_SIZE = 100
NUM_EPOCHS = 2
LEARNING_RATE = 0.003

FILE_PATH = '../../../data/MNIST/'

if not os.path.exists(FILE_PATH):
    if not os.path.exists('../../../data'):
        os.mkdir('../../../data')
    # 下载数据集
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

# 加载MNIST数据集
train_dataset = mindspore.dataset.MnistDataset(
    dataset_dir=FILE_PATH,
    usage='train',
    shuffle=True
).map(operations=image_transforms, input_columns="image").batch(batch_size=BATCH_SIZE)

test_dataset = mindspore.dataset.MnistDataset(
    dataset_dir=FILE_PATH,
    usage='test',
    shuffle=False
).map(operations=image_transforms, input_columns="image").batch(batch_size=BATCH_SIZE)


class BiRNN(nn.Cell):
    """双向循环神经网络"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 双向LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Dense(hidden_size * 2, num_classes, weight_init=HeUniform(math.sqrt(5)))

    def construct(self, x):
        # 设置初始状态
        hidden0 = ops.zeros(size=(self.num_layers * 2, x.shape[0], self.hidden_size))
        cell0 = ops.zeros(size=(self.num_layers * 2, x.shape[0], self.hidden_size))

        out, _ = self.lstm(x, (hidden0, cell0))

        out = self.linear(out[:, -1, :])
        return out


model = BiRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = nn.optim.Adam(model.trainable_params(), LEARNING_RATE)

# 绑定损失函数
train_model = nn.WithLossCell(model, loss_fn=criterion)
train_model = nn.TrainOneStepCell(train_model, optimizer)

# 训练
for epoch in range(NUM_EPOCHS):
    for i, (image, label) in enumerate(train_dataset.create_tuple_iterator()):
        image = ops.reshape(image, (-1, SEQUENCE_LENGTH, INPUT_SIZE))
        total_step = train_dataset.get_dataset_size()
        train_model.set_train()
        label = mindspore.Tensor(label, mstype.int32)
        loss = train_model(image, label)

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{total_step}], Loss: {loss.asnumpy().item():.4f}')

# 测试模型
model.set_train(False)
CORRECT = 0
TOTAL = 0
for image, label in test_dataset.create_tuple_iterator():
    image = ops.reshape(image, (-1, SEQUENCE_LENGTH, INPUT_SIZE))
    label = mindspore.Tensor(label, mstype.int32)
    outputs = model(image)
    _, predicted = ops.max(outputs.value(), 1)
    TOTAL += label.shape[0]
    CORRECT += (predicted == label).sum().asnumpy().item()

print(f'Test Accuracy of the model on the 10000 test images: {(100 * CORRECT / TOTAL):.2f} %')

# 保存模型
SAVE_PATH = './rnn.ckpt'
mindspore.save_checkpoint(model, SAVE_PATH)

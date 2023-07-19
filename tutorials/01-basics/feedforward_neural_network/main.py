# 超参数
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

INPUT_SIZE = 784
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
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                  't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        url = (base_url + file_name).format(**locals())
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

# 带一个隐藏层的全连接神经网络
class NeuralNet(nn.Cell):
    def __init__(self,input_size, hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.fc1=nn.Dense(input_size,hidden_size,weight_init=HeUniform(math.sqrt(5)))
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(hidden_size,num_classes,weight_init=HeUniform(math.sqrt(5)))

    def construct(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(INPUT_SIZE, hidden_size, num_classes)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = nn.optim.Adam(model.trainable_params(),learning_rate)

# 绑定训练参数
model_with_loss = nn.WithLossCell(model,criterion)
train_model = nn.TrainOneStepCell(model_with_loss,optimizer)

# 训练模型
for epoch in range(num_epochs):
    for i, (image, label) in enumerate(train_dataset.create_tuple_iterator()):
        total_step = train_dataset.get_dataset_size()
        train_model.set_train()
        image = ops.reshape(image, (-1, 28*28))
        label = mindspore.Tensor(label, mstype.int32)
        loss = train_model(image, label)
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.asnumpy().item()))

model.set_train(False)

# 测试模型
correct = 0
total = 0
for image, label in test_dataset.create_tuple_iterator():
    label = mindspore.Tensor(label, mstype.int32)
    image = ops.reshape(image, (-1, 28*28))
    outputs = model(image)
    _, predicted = ops.max(outputs.value(), 1)
    total += label.shape[0]
    correct += (predicted == label).sum().asnumpy().item()

print('Test Accuracy of the model on the 10000 test images: {:.2f} %'.format(100 * correct / total))

# Save the model checkpoint
save_path = './model.ckpt'
mindspore.save_checkpoint(model, save_path)


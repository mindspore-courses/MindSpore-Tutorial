"""MindInsight"""
import gzip
import math
import os
import shutil
from urllib import request

import mindspore
import numpy as np
from mindspore import nn, ops, SummaryRecord
from mindspore.common.initializer import HeUniform
from mindspore.dataset.vision import transforms
import mindspore.common.dtype as mstype

input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.0001

mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
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

class NeuralNet(nn.Cell):
    """带一个隐藏层的全连接神经网络"""
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Dense(input_size, hidden_size, weight_init=HeUniform(math.sqrt(5)))
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(hidden_size, num_classes, weight_init=HeUniform(math.sqrt(5)))

    def construct(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = nn.optim.Adam(model.trainable_params(), learning_rate)

def forward(images, labels):
    """正向传播"""
    output = model(images)
    loss = criterion(output, labels)
    return loss, output


# 求梯度
grad_fn = ops.value_and_grad(forward, None, optimizer.trainable_params, has_aux=True)


def main():
    """主函数"""
    # 训练模型
    with SummaryRecord('./summary', network=model) as summary_record:
        for epoch in range(num_epochs):
            for i, (image, label) in enumerate(train_dataset.create_tuple_iterator()):
                total_step = train_dataset.get_dataset_size()
                model.set_train()
                image = image.view(image.shape[0], -1)
                label = mindspore.Tensor(label, mstype.int32)

                (loss, output), grads = grad_fn(image, label)
                optimizer(grads)

                _, argmax = ops.max(output, 1)
                accuracy = (label == argmax.squeeze()).float().asnumpy().mean()
                if (i + 1) % 100 == 0:
                    print(f'Step [{i + 1}/{total_step}], Loss: {loss.asnumpy().item():.4f}, Acc: {accuracy.item():.2f}')
                    summary_record.add_value('scalar', 'loss', loss)
                    for tag, value in model.parameters_and_names():
                        tag = tag.replace('.', '/')
                        summary_record.add_value('histogram', tag, value.data)
                    print(image.view(-1, 28, 28)[0].shape)
                    summary_record.record(epoch * 600 + i)


if __name__ == '__main__':
    main()

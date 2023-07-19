"""深度残差网络"""
import math
import os
import tarfile
import urllib.request

import mindspore.common.dtype as mstype
import mindspore.dataset.vision
from mindspore.dataset.vision import transforms
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore.common.initializer import HeUniform

FILE_PATH = '../../../data/CIFAR-10'

if not os.path.exists(FILE_PATH):
    if not os.path.exists('../../../data'):
        os.mkdir('../../../data')
    # 下载CIFAR-10数据集
    os.mkdir(FILE_PATH)
    URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    FILE_NAME = 'cifar-10-binary.tar.gz'
    print("正在从" + URL + "下载CIFAR-10数据集...")
    result = urllib.request.urlretrieve(URL, os.path.join(FILE_PATH, FILE_NAME))
    with tarfile.open(os.path.join(FILE_PATH, FILE_NAME), 'r:gz') as tar:
        print("正在解压数据集...")
        for member in tar.getmembers():
            if member.name.startswith('cifar-10-batches-bin'):
                member.name = os.path.basename(member.name)
                tar.extract(member, path=FILE_PATH)
    os.remove(os.path.join(FILE_PATH, FILE_NAME))

# 超参数
NUM_EPOCHS = 80
BATCH_SIZE = 100
LEARNING_RATE = 0.001

# 预处理
data_transforms = [
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()]

# 导入CIFAR-10数据集
train_dataset = mindspore.dataset.Cifar10Dataset(
    dataset_dir=FILE_PATH,
    usage='train',
    shuffle=True
).map(operations=data_transforms, input_columns="image").batch(batch_size=BATCH_SIZE)

test_dataset = mindspore.dataset.Cifar10Dataset(
    dataset_dir=FILE_PATH,
    usage='test',
    shuffle=False
).map(operations=transforms.ToTensor()).batch(batch_size=BATCH_SIZE)


def conv3x3(in_channels, out_channels, stride=1):
    """3x3卷积核"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, pad_mode='pad')


class ResidualBlock(nn.Cell):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def construct(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Cell):
    """残差网络"""
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.batch_norm = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.linear = nn.Dense(64, num_classes, weight_init=HeUniform(math.sqrt(5)))

    def make_layer(self, block, out_channels, blocks, stride=1):
        """创建层"""
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.SequentialCell(
                conv3x3(self.in_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.SequentialCell(*layers)

    def construct(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(ops.shape(out)[0], -1)
        out = self.linear(out)
        return out


model = ResNet(ResidualBlock, [2, 2, 2])

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = nn.optim.Adam(model.trainable_params(), LEARNING_RATE)
# 绑定损失函数
train_model = nn.WithLossCell(model, loss_fn=criterion)

train_model = nn.TrainOneStepCell(train_model, optimizer)

# 训练
CURR_LR = LEARNING_RATE
for epoch in range(NUM_EPOCHS):
    for j, (image, label) in enumerate(train_dataset.create_tuple_iterator()):
        total_step = train_dataset.get_dataset_size()
        train_model.set_train()
        label = mindspore.Tensor(label, mstype.int32)
        loss = train_model(image, label)

        if (j + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{j + 1}/{total_step}], Loss: {loss.asnumpy().item():.4f}')

    # 调整学习率
    if (epoch + 1) % 20 == 0:
        CURR_LR /= 3
        ops.assign(optimizer.learning_rate, Tensor(CURR_LR))
        print(f"Current Leaning Rate:{optimizer.get_lr().asnumpy().item()}")

# 测试模型
model.set_train(False)
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
SAVE_PATH = './resnet.ckpt'
mindspore.save_checkpoint(model, SAVE_PATH)

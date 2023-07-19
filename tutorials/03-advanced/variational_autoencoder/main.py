"""VAE"""
import gzip
import math
import os
import shutil
import urllib.request

import mindspore
from mindspore import nn
from mindspore.common.initializer import HeUniform
from mindspore.dataset.vision import transforms
from mindspore import ops
from img_utils import to_image

SAMPLE_DIR = 'samples'
if not os.path.exists(SAMPLE_DIR):
    os.makedirs(SAMPLE_DIR)

# 超参数
IMAGE_SIZE = 784
H_DIM = 400
Z_DIM = 20
NUM_EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# MNIST数据集
FILE_PATH = "../../../data/MNIST/"

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

# 创建文件夹
if not os.path.exists(SAMPLE_DIR):
    os.makedirs(SAMPLE_DIR)

# Image processing
# transform = [
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
#                                      std=(0.5, 0.5, 0.5))]

dataset = mindspore.dataset.MnistDataset(
    dataset_dir=FILE_PATH,
    usage='train',
).map(operations=transforms.ToTensor(), input_columns="image").batch(BATCH_SIZE)


class VAE(nn.Cell):
    """VAE模型"""

    def __init__(self, img_size=784, h_dim=400, z_dim=20):
        super().__init__()
        self.fc1 = nn.Dense(img_size, h_dim, weight_init=HeUniform(math.sqrt(5)))
        self.fc2 = nn.Dense(h_dim, z_dim, weight_init=HeUniform(math.sqrt(5)))
        self.fc3 = nn.Dense(h_dim, z_dim, weight_init=HeUniform(math.sqrt(5)))
        self.fc4 = nn.Dense(z_dim, h_dim, weight_init=HeUniform(math.sqrt(5)))
        self.fc5 = nn.Dense(h_dim, img_size, weight_init=HeUniform(math.sqrt(5)))

    def encode(self, _x):
        """编码"""
        h = ops.relu(self.fc1(_x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, _mu, log_var):
        """重置参数"""
        std = ops.exp(log_var / 2)
        eps = ops.randn_like(std)
        return _mu + eps * std

    def decode(self, _z):
        """解码"""
        h = ops.relu(self.fc4(_z))
        return ops.sigmoid(self.fc5(h))

    def construct(self, _x):
        """前向传播"""
        _mu, log_var = self.encode(_x)
        _z = self.reparameterize(_mu, log_var)
        x_reconst = self.decode(_z)
        return x_reconst, _mu, log_var


model = VAE()
optimizer = nn.optim.Adam(model.trainable_params(), LEARNING_RATE)


def forward(_x):
    """前向传播"""
    x_reconst, _mu, log_var = model(_x)

    _reconst_loss = ops.binary_cross_entropy(x_reconst, _x, reduction='sum')
    _kl_div = - 0.5 * ops.sum(1 + log_var - _mu.pow(2) - log_var.exp())
    return _reconst_loss, _kl_div


grad_fn = ops.value_and_grad(forward, None, optimizer.parameters)

# 开始训练
for epoch in range(NUM_EPOCHS):
    X = None
    for i, (X, _) in enumerate(dataset.create_tuple_iterator()):
        model.set_train()
        total_step = dataset.get_dataset_size()
        X = X.view(-1, IMAGE_SIZE)
        (reconst_loss, kl_div), grads = grad_fn(X)
        optimizer(grads)

        if (i + 1) % 10 == 0:
            print(f"Epoch[{epoch + 1}/{NUM_EPOCHS}], "
                  f"Step [{i + 1}/{total_step}], "
                  f"Reconst Loss: {reconst_loss.asnumpy().item():.4f}, "
                  f"KL Div: {kl_div.asnumpy().item():.4f}")

    # Save the sampled images
    z = ops.randn(BATCH_SIZE, Z_DIM)
    out = model.decode(z).view(-1, 1, 28, 28)
    to_image(out, os.path.join(SAMPLE_DIR, f'sampled-{epoch + 1}.png'))

    # Save the reconstructed images
    out, _, _ = model(X)
    x_concat = ops.cat([X.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], axis=3)
    to_image(x_concat, os.path.join(SAMPLE_DIR, f'reconst-{epoch + 1}.png'))

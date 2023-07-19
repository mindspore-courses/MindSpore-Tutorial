"""生成对抗网络"""
import gzip
import math
import os
import shutil
import urllib.request

import mindspore
from mindspore import nn, ops
from mindspore.common.initializer import HeUniform
from mindspore.dataset.vision import transforms
from img_utils import to_image

# 超参数
LATENT_SIZE = 64
HIDDEN_SIZE = 256
IMAGE_SIZE = 784
NUM_EPOCHS = 200
BATCH_SIZE = 100
SAMPLE_DIR = 'samples'

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
transform = [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],  # 1 for greyscale channels
                         std=[0.5])]

dataset = mindspore.dataset.MnistDataset(
    dataset_dir=FILE_PATH,
    usage='train',
).map(operations=transform, input_columns="image").batch(BATCH_SIZE)

D = nn.SequentialCell(
    nn.Dense(IMAGE_SIZE, HIDDEN_SIZE, weight_init=HeUniform(math.sqrt(5))),
    nn.LeakyReLU(0.2),
    nn.Dense(HIDDEN_SIZE, HIDDEN_SIZE, weight_init=HeUniform(math.sqrt(5))),
    nn.LeakyReLU(0.2),
    nn.Dense(HIDDEN_SIZE, 1, weight_init=HeUniform(math.sqrt(5))),
    nn.Sigmoid()
)

G = nn.SequentialCell(
    nn.Dense(LATENT_SIZE, HIDDEN_SIZE, weight_init=HeUniform(math.sqrt(5))),
    nn.ReLU(),
    nn.Dense(HIDDEN_SIZE, HIDDEN_SIZE, weight_init=HeUniform(math.sqrt(5))),
    nn.ReLU(),
    nn.Dense(HIDDEN_SIZE, IMAGE_SIZE, weight_init=HeUniform(math.sqrt(5))),
    nn.Tanh()
)

# 损失函数与优化器
criterion = nn.BCELoss()
D_Optim = nn.optim.Adam(D.trainable_params(), learning_rate=0.0002)
G_Optim = nn.optim.Adam(G.trainable_params(), learning_rate=0.0002)


def denorm(x):
    """反正则化"""
    out = (x + 1) / 2
    return ops.clamp(out, 0, 1)


def g_forward(valid):
    """生成器前向传播"""
    # z = ops.StandardNormal()((real_imgs.shape[0], latent_size))
    _z = ops.randn(BATCH_SIZE, LATENT_SIZE)
    gen_imgs = G(_z)
    _g_loss = criterion(D(gen_imgs), valid)
    return _g_loss, gen_imgs


def d_forward(_real_imgs, _gen_imgs, _valid, _fake):
    """判别器前向传播"""
    _real_score = D(_real_imgs)
    _fake_score = D(_gen_imgs)
    real_loss = criterion(_real_score, _valid)
    fake_loss = criterion(_fake_score, _fake)
    _d_loss = real_loss + fake_loss
    return _d_loss, _real_score, _fake_score


grad_g = ops.value_and_grad(g_forward, None, G_Optim.parameters, has_aux=True)
grad_d = ops.value_and_grad(d_forward, None, D_Optim.parameters, has_aux=True)

for epoch in range(NUM_EPOCHS):
    IMAGE = None
    FAKE_IMAGES = None
    for i, (IMAGE, _) in enumerate(dataset.create_tuple_iterator()):
        total_step = dataset.get_dataset_size()
        G.set_train()
        D.set_train()
        IMAGE = ops.reshape(IMAGE, (BATCH_SIZE, -1))

        real_labels = ops.ones((BATCH_SIZE, 1))
        fake_labels = ops.zeros((BATCH_SIZE, 1))

        z = ops.randn(BATCH_SIZE, LATENT_SIZE)

        (d_loss, real_score, fake_score), d_grads = grad_d(IMAGE, G(z), real_labels, fake_labels)
        D_Optim(d_grads)

        # Generator
        (g_loss, FAKE_IMAGES), g_grads = grad_g(real_labels)
        G_Optim(g_grads)

        if (i + 1) % 200 == 0:
            print(f'Epoch [{epoch}/{NUM_EPOCHS}], Step [{i + 1}/{total_step}], '
                  f'd_loss: {d_loss.asnumpy().item():.4f}, g_loss: {g_loss.asnumpy().item():.4f}, '
                  f'D(x): {ops.mean(real_score).asnumpy().item():.2f}, D(G(z)): {ops.mean(fake_score).asnumpy().item():.2f}')

    # Save real images
    if (epoch + 1) == 1:
        IMAGE = ops.reshape(IMAGE, (IMAGE.shape[0], 1, 28, 28))
        to_image(denorm(IMAGE), os.path.join(SAMPLE_DIR, 'real_images.png'))

    # Save sampled images
    FAKE_IMAGES = ops.reshape(FAKE_IMAGES, (FAKE_IMAGES.shape[0], 1, 28, 28))
    to_image(denorm(FAKE_IMAGES), os.path.join(SAMPLE_DIR, F'fake_images-{epoch + 1}.png'))

mindspore.save_checkpoint(G, './g.ckpt')
mindspore.save_checkpoint(D, './d.ckpt')

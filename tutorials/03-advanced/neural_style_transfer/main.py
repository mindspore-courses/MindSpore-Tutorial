"""风格转换"""
from __future__ import division

import argparse

import mindspore
import mindspore.dataset.transforms as trans
import numpy as np
from PIL import Image
from mindcv.models import vgg19
from mindspore import nn, ops, Parameter
from mindspore.dataset.transforms import Compose

from img_utils import to_image


def load_image(image_path, transform=None, max_size=None, shape=None):
    """载入图片"""
    image = Image.open(image_path)

    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)

    if shape:
        image = image.resize(shape, Image.LANCZOS)

    if transform:
        image = transform(image)
        image = mindspore.Tensor(image)
        # image = image.unsqueeze(0)
    return image


class VGGNet(nn.Cell):
    """VGG19"""
    def __init__(self):
        super().__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = vgg19(pretrained=True).features

    def construct(self, x):
        features = []
        cell_list = self.vgg.cell_list
        i = 0
        y = -1
        for layer in cell_list:
            x = layer(x)
            if str(i) in self.select:
                features.append(x)
                y += 1
            features[y] = x
            i += 1
        return features


vgg = VGGNet()


def forward(content, target, style):
    """前向传播"""
    content_features = vgg(content)
    target_features = vgg(target)
    style_features = vgg(style)

    style_loss = 0
    content_loss = 0
    for f1_, f2_, f3_ in zip(target_features, content_features, style_features):
        content_loss += ops.mean((f1_ - f2_) ** 2)

        # Reshape convolutional feature maps
        c, h, w = f1_.shape[1], f1_.shape[2], f1_.shape[3]
        f1_ = f1_.view(c, h * w)
        f3_ = f3_.view(c, h * w)

        # Compute gram matrix
        f1_ = ops.matmul(f1_, f1_.t())
        f3_ = ops.matmul(f3_, f3_.t())

        # Compute style loss with target and style images
        style_loss += ops.mean((f1_ - f3_) ** 2) / (c * h * w)
    return content_loss + config.style_weight * style_loss, content_loss, style_loss


def main(_config):
    """主函数"""
    # 图像预处理
    transforms = Compose([trans.vision.ToTensor()])
                          # trans.vision.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_hwc=False)])

    content = load_image(_config.content, transforms, max_size=_config.max_size)
    style = load_image(_config.style, transforms, shape=[content.shape[2], content.shape[3]])

    # 初始化目标图像
    target = Parameter(content, requires_grad=True)

    optimizer = nn.optim.Adam([target], learning_rate=_config.lr, beta1=0.5, beta2=0.999)
    # content, target, style,
    # position is 1
    grad_fn = ops.value_and_grad(forward, 1, has_aux=True)
    vgg.set_train(False)

    for step in range(_config.total_step):
        (_, content_loss, style_loss), grads = grad_fn(content, target, style)
        optimizer((grads,))
        if (step + 1) % _config.log_step == 0:
            print(f'Step [{step + 1}/{_config.total_step}], '
                  f'Content Loss: {content_loss.asnumpy().item():.4f}, '
                  f'Style Loss: {style_loss.asnumpy().item():.4f}')

        if (step + 1) % _config.sample_step == 0:
            # Save the generated image
            # denorm = trans.vision.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = target.clone().squeeze()
            img = img.clamp(0, 1)
            to_image(img, f'output-{step + 1}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='png/content.png')
    parser.add_argument('--style', type=str, default='png/style.png')
    parser.add_argument('--max_size', type=int, default=400)
    parser.add_argument('--total_step', type=int, default=2000)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--style_weight', type=float, default=100)
    parser.add_argument('--lr', type=float, default=0.003)
    config = parser.parse_args()
    print(config)
    main(config)

from __future__ import division

import argparse

import mindspore
import mindspore.common.dtype as mstype
import mindspore.dataset.vision.py_transforms as pvision
import numpy as np
from PIL import Image
from mindcv.models import vgg19
from mindspore import nn, ops, Tensor, Parameter

from img_utils import to_image


def load_image(image_path, transform=None, max_size=None, shape=None):
    """Load an image and convert it to a torch tensor."""
    image = Image.open(image_path)

    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)

    if shape:
        image = image.resize(shape, Image.LANCZOS)

    if transform:
        image = transform[0](image)
        image = transform[1](image)
        image = mindspore.Tensor(image)
        image = ops.unsqueeze(image, 0)
    return image


class VGGNet(nn.Cell):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = vgg19(pretrained=True).features

    def construct(self, x):
        features = []
        cell_list = self.vgg.cell_list
        i = 0
        for layer in cell_list:
            x = layer(x)
            if str(i) in self.select:
                features.append(x)
            i += 1
        return features


vgg = VGGNet()


def forward(content, target, style):
    content_features = vgg(content)
    target_features = vgg(target)
    style_features = vgg(style)

    style_loss = 0
    content_loss = 0
    for f1, f2, f3 in zip(target_features, content_features, style_features):
        # Compute content loss with target and content images
        content_loss += ops.mean((f1 - f2) ** 2)

        # Reshape convolutional feature maps
        c, h, w = f1.shape[1], f1.shape[2], f1.shape[3]
        f1 = f1.view(c, h * w)
        f3 = f3.view(c, h * w)

        # Compute gram matrix
        f1 = ops.matmul(f1, f1.t())
        f3 = ops.matmul(f3, f3.t())

        # Compute style loss with target and style images
        style_loss += ops.mean((f1 - f3) ** 2) / (c * h * w)
    return content_loss + config.style_weight * style_loss, content_loss, style_loss


def main(config):
    # 图像预处理
    transforms = [pvision.ToTensor(), pvision.Normalize(mean=(0.485, 0.456, 0.406),
                                                        std=(0.229, 0.224, 0.225))]

    content = load_image(config.content, transforms, max_size=config.max_size)
    style = load_image(config.style, transforms, shape=[content.shape[2], content.shape[3]])

    # 初始化目标图像
    target = Parameter(content, requires_grad=True)

    optimizer = nn.optim.Adam([target], learning_rate=config.lr, beta1=0.5, beta2=0.999)
    grad_fn = ops.value_and_grad(forward, None, optimizer.parameters, has_aux=True)
    vgg.set_train(False)

    for step in range(config.total_step):
        (loss, content_loss, style_loss), grads = grad_fn(content, target, style)
        optimizer(grads)
        target = vgg.trainable_params()[0]
        if (step + 1) % config.log_step == 0:
            print('Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}'
                  .format(step + 1, config.total_step, content_loss.asnumpy().item(), style_loss.asnumpy().item()))

        if (step + 1) % config.sample_step == 0:
            # Save the generated image
            denorm = pvision.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
            img = target.clone().squeeze()
            img = denorm(img).clamp_(0, 1)
            to_image(img, 'output-{}.png'.format(step + 1))


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

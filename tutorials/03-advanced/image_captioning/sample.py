"""测试"""
import argparse
import json

import matplotlib.pyplot as plt
import mindspore
import numpy as np
from PIL import Image
from mindspore import Tensor
from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import transforms

from model import EncoderCNN, DecoderRNN


def load_image(image_path, transform=None):
    """加载图片"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize([256, 256], Image.LANCZOS)

    if transform is not None:
        image = transform(image)
        image = Tensor(image[0])
        image = image.unsqueeze(0)

    return image


def main(_args):
    """主函数"""
    # Image preprocessing
    transform = Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225), is_hwc=False)])

    # Load vocabulary wrapper
    with open(_args.json_path, 'rb') as f:
        vocab = json.load(f)

    # Build models
    encoder = EncoderCNN(_args.EMBED_SIZE)  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(_args.EMBED_SIZE, _args.HIDDEN_SIZE, len(vocab), _args.NUM_LAYERS)

    encoder.set_train(False)
    decoder.set_train(False)

    # Load the trained model parameters
    mindspore.load_param_into_net(encoder, mindspore.load_checkpoint(_args.encoder_path))
    mindspore.load_param_into_net(decoder, mindspore.load_checkpoint(_args.decoder_path))

    # Prepare an image
    image = load_image(_args.image, transform)
    image_tensor = mindspore.Tensor(image)

    # Generate an caption from the image
    # image_tensor = ops.ones(image_tensor.shape)
    feature = encoder(image_tensor)

    print(feature)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].asnumpy()  # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = list(vocab.keys())[list(vocab.values()).index(word_id)]
        # word = vocab[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    # Print out the image and the generated caption
    print(sentence)
    image = Image.open(_args.image)
    plt.imshow(np.asarray(image))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-1-100.ckpt',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-1-100.ckpt',
                        help='path for trained decoder')
    parser.add_argument('--json_path', type=str,
                        default='../../../data/COCO/mindrecord/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)

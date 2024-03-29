"""看图说话"""
import argparse
import json
import os

import mindspore.common.dtype as mstype
import mindspore.dataset
import numpy as np
from mindspore import nn, ops

from dataset import create_dataset
from model import EncoderCNN, DecoderRNN


def main(_args):
    """主函数"""

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    with open(args.vocab_path, 'rb') as f:
        vocab = json.load(f)

    dataset = create_dataset(data_path=args.image_dir, crop_size=args.crop_size, batch_size=128)

    encoder = EncoderCNN(args.embed_size)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder.update_parameters_name("encoder")
    decoder.update_parameters_name("decoder")

    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    encoder.fine_tune(False)

    params = list(decoder.trainable_params()) + list(encoder.linear.trainable_params()) + list(
        encoder.batch_norm.trainable_params())
    optimizer = nn.optim.Adam(params, args.learning_rate)

    def forward(images, captions, lengths):
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        # captions = Tensor(captions, dtype=mstype.float32)
        outputs = outputs.swapaxes(1, 2)
        print("output")
        print(outputs)
        print("cap")
        print(captions)
        loss = criterion(outputs, captions)
        return loss

    grad_fn = ops.value_and_grad(forward, None, optimizer.parameters)
    total_step = dataset.get_dataset_size()
    encoder.set_train()
    decoder.set_train()

    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(dataset.create_tuple_iterator()):
            # captions = Tensor(captions, dtype=mstype.float32)
            max_len = max(lengths).asnumpy().item()
            new_cap = ops.zeros(size=(ops.shape(captions)[0], max_len), dtype=mstype.int32)
            for i, caption in enumerate(captions):
                new_cap[i] = caption[:max_len]
            loss, grads = grad_fn(images, new_cap, lengths)
            optimizer(grads)

            # Print log info
            if i % args.log_step == 0:
                print(f'Epoch [{epoch}/{ args.num_epochs}], Step [{i}/{total_step}], '
                      f'Loss: {loss.asnumpy().item():.4f}, '
                      f'Perplexity: {np.exp(loss.asnumpy().item()):5.4f}')

                # Save the model checkpoints
            if (i + 1) % args.save_step == 0:
                mindspore.save_checkpoint(decoder, os.path.join(
                    args.model_path, f'decoder-{epoch + 1}-{i + 1}.ckpt'))
                mindspore.save_checkpoint(encoder, os.path.join(
                    args.model_path, f'encoder-{epoch + 1}-{i + 1}.ckpt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=256, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str,
                        default='../../../data/COCO/mindrecord/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str,
                        default='../../../data/COCO/mindrecord/TRAIN_IMAGES_coco_5_cap_per_img_5_min_word_freq.mindrecord',
                        help='directory for resized images')
    # parser.add_argument('--caption_path', type=str, default='../../../data/COCO/annotations/captions_train2014.json',
    #                     help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)

import argparse
import os
import pickle

import mindspore.dataset
import numpy as np
from mindspore import nn, ops
import mindspore.dataset.vision.py_transforms as pvision
from data_loader import get_dataset
from model import EncoderCNN, DecoderRNN
from build_vocab import Vocabulary


def forward(images, captions, lengths, encoder, decoder, criterion):
    features = encoder(images)
    outputs = decoder(features, captions, lengths)
    loss = criterion(outputs, captions)

    return loss


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = [
        pvision.RandomCrop(args.crop_size),
        pvision.RandomHorizontalFlip(),
        pvision.ToTensor(),
        pvision.Normalize((0.485, 0.456, 0.406),
                          (0.229, 0.224, 0.225))]

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    dataset = get_dataset(args.image_dir, args.caption_path, vocab,
                          transform, args.batch_size,
                          shuffle=True, python_multiprocessing=True)

    encoder = EncoderCNN(args.embed_size)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)

    criterion = nn.CrossEntropyLoss()
    cell_list = nn.CellList()
    cell_list.append(decoder)
    cell_list.append(encoder)
    cell_list.append(encoder.bn)
    # o_params = list(decoder.trainable_params()) + list(encoder.trainable_params()) + list(encoder.bn.trainable_params())
    # o_params[5].name = 'linear1_weight'
    # o_params[6].name = 'linear1_bias'
    optimizer = nn.optim.Adam(cell_list.trainable_params(), args.learning_rate)
    grad_fn = ops.value_and_grad(forward, None, optimizer.parameters)
    total_step = dataset.get_dataset_size()
    encoder.set_train()
    decoder.set_train()
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(dataset.create_dict_iterator()):
            loss, grads = grad_fn(images, captions, lengths, encoder, decoder, criterion)
            optimizer(grads)

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.asnumpy().item(),
                              np.exp(loss.asnumpy().item())))

                # Save the model checkpoints
            if (i + 1) % args.save_step == 0:
                mindspore.save_checkpoint(decoder, os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                mindspore.save_checkpoint(encoder, os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='../../../data/COCO/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='../../../data/COCO/resized2014',
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='../../../data/COCO/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)

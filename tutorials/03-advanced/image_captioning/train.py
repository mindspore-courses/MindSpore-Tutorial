import os
import pickle

import mindspore.dataset
import numpy as np
from mindspore import nn, ops
from mindspore.dataset.vision import transforms
from mindspore.dataset.transforms import Compose
from data_loader import get_dataset
from model import EncoderCNN, DecoderRNN


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
    transform = Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    dataset = get_dataset(args.image_dir, args.caption_path, vocab,
                          transform, args.batch_size,
                          shuffle=True, python_multiprocessing=True)

    encoder = EncoderCNN(args.embed_size)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)

    criterion = nn.CrossEntropyLoss()
    params = list(decoder.trainable_params()) + list(encoder.trainable_params()) + list(encoder.bn.trainable_params())
    optimizer = nn.optim.Adam(params, args.learning_rate)
    grad_fn = ops.value_and_grad(forward, None, optimizer.parameters)
    total_step = len(dataset)
    encoder.set_train()
    decoder.set_train()
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(dataset):
            loss, grads = grad_fn(images, captions, lengths, encoder, decoder, criterion)
            optimizer(grads)

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.asnumpy().item(), np.exp(loss.asnumpy().item())))

                # Save the model checkpoints
            if (i + 1) % args.save_step == 0:
                mindspore.save_checkpoint(decoder, os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                mindspore.save_checkpoint(encoder, os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))

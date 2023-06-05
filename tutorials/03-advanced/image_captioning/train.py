import os
import pickle

import mindspore.dataset
from mindspore import nn
from mindspore.dataset.vision import transforms
from mindspore.dataset.transforms import Compose
from data_loader import get_dataset
from model import EncoderCNN, DecoderRNN


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
    

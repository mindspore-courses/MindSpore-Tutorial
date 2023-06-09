import math

import mindspore
import mindspore.nn as nn
import mindcv.models as models
from mindspore import ops, Tensor
from mindspore.common.initializer import HeUniform


class EncoderCNN(nn.Cell):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.cells())[:-1]
        self.resnet = nn.SequentialCell(*modules)
        self.linear = nn.Dense(resnet.classifier.in_channels, embed_size, weight_init=HeUniform(math.sqrt(5)))
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def construct(self, images):
        features = self.resnet(images)
        ops.stop_gradient(features)
        features = features.reshape(features.shape[0], -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Cell):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Dense(hidden_size, vocab_size, weight_init=HeUniform(math.sqrt(5)))
        self.max_seq_length = max_seq_length

    def construct(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = ops.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings, seq_length=lengths)
        outputs = self.linear(hiddens)
        return outputs[:, 1:, :]

    def sample(self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = ops.stack(sampled_ids, 1)
        return sampled_ids

import math

import mindspore.dataset.vision
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor
from mindspore.common.initializer import HeUniform, _calculate_fan_in_and_fan_out, initializer, Uniform
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

from data_utils import Corpus

# 超参数
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 5
num_samples = 1000  # 要采样的词数
batch_size = 20
seq_length = 30
learning_rate = 0.002
GRADIENT_CLIP_MIN = -64000
GRADIENT_CLIP_MAX = 64000

# 加载数据集
corpus = Corpus()
ids = corpus.get_data('../../../data/PennTreeBank/ptb.train.txt', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.shape[1] // seq_length


# pytorch like style
class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True, activation=None):
        super().__init__(in_channels, out_channels, weight_init='normal', bias_init='zeros', has_bias=has_bias,
                         activation=activation)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))


# 基于RNN的语言模型
class RNNLM(nn.Cell):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = Dense(hidden_size, vocab_size)

    def construct(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)
        out = ops.reshape(out, (out.shape[0] * out.shape[1], out.shape[2]))
        out = self.linear(out)
        return out, (h, c)


model = RNNLM(vocab_size, embed_size, hidden_size, num_layers)


def forward(inputs, states, targets):
    # states = tuple(ops.stop_gradient(state) for state in states)
    outputs, states = model(inputs, states)
    loss = criterion(outputs, ops.reshape(targets, Tensor(np.array([1]))))
    return loss


# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = nn.optim.Adam(model.trainable_params(), learning_rate)
grad_fn = ops.value_and_grad(forward, None, optimizer.parameters)

# 训练
for epoch in range(num_epochs):
    model.set_train()
    states = (ops.zeros((num_layers, batch_size, hidden_size)),
              ops.zeros((num_layers, batch_size, hidden_size)))

    for i in range(0, ids.shape[1] - seq_length, seq_length):
        inputs = ids[:, i:i + seq_length]
        targets = mindspore.Tensor.int(ids[:, (i + 1):(i + 1) + seq_length])

        loss, grads = grad_fn(inputs, states, targets)
        grads = ops.clip_by_global_norm(grads, 0.5)
        optimizer(grads)

        step = (i + 1) // seq_length
        if step % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                  .format(epoch + 1,
                          num_epochs,
                          step,
                          num_batches,
                          loss.asnumpy().item(),
                          np.exp(loss.asnumpy().item())))

# 测试模型
model.set_train(False)
with open('sample.txt', 'w') as f:
    state = (ops.zeros((num_layers, 1, hidden_size)),
             ops.zeros((num_layers, 1, hidden_size)))

    prob = ops.ones(vocab_size)
    input = ops.multinomial(prob, num_samples=1).unsqueeze(1)

    for i in range(num_samples):
        output, state = model(input, state)

        prob = output.exp()
        word_id = ops.multinomial(prob, num_samples=1).asnumpy().item()

        input = ops.fill(mstype.int32, input.shape, 1)

        word = corpus.dictionary.idx2word[word_id]
        word = '\n' if word == '<eos>' else word + ' '
        f.write(word)

        if (i + 1) % 100 == 0:
            print('Sampled [{}/{}] words and save to {}'.format(i + 1, num_samples, 'sample.txt'))

# 保存模型
save_path = './lm.ckpt'
mindspore.save_checkpoint(model, save_path)

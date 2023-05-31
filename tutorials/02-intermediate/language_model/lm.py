import os
import urllib.request
import zipfile

import mindspore.common.dtype as mstype
import mindspore.dataset.vision
import mindspore.dataset.vision.transforms as transforms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from du import Corpus

# 超参数
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 5
num_samples = 1000  # 要采样的词数
batch_size = 20
seq_length = 30
learning_rate = 0.002

# 加载数据集
file_path = '../../data/PennTreeBank/'

image_transforms = transforms.ToTensor()
label_transforms = transforms.ToTensor(output_type=np.int32)

# 加载MNIST数据集
train_dataset = mindspore.dataset.PennTreebankDataset(
    dataset_dir=file_path,
    usage='train',
    shuffle=True
).map(operations=image_transforms, input_columns="image").batch(batch_size=batch_size)

test_dataset = mindspore.dataset.PennTreebankDataset(
    dataset_dir=file_path,
    usage='test',
    shuffle=False
).map(operations=image_transforms, input_columns="image").batch(batch_size=batch_size)

# 基于RNN的语言模型
class RNNLM(nn.Cell):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Dense(hidden_size, vocab_size)

    def construct(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)
        out = ops.reshape(out, (out.shape[0] * out.shape[1], out.shape[2]))
        out = self.linear(out)
        return out, (h, c)


model = RNNLM(vocab_size, embed_size, hidden_size, num_layers)


def forward(inputs, states, targets):
    states = ops.stop_gradient(states)
    outputs, states = model(inputs, states)
    loss = criterion(outputs, ops.reshape(targets, -1))
    return loss


# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = nn.optim.Adam(model.trainable_params(), learning_rate)
grad_fn = ops.value_and_grad(forward, None, optimizer.parameters)

# 训练
for epoch in range(num_epochs):
    model.set_train()
    states = mindspore.Parameter((ops.zeros((num_layers, batch_size, hidden_size)),
                                  ops.zeros((num_layers, batch_size, hidden_size))), requires_grad=False)

    for i in range(0, ids.shape[1] - seq_length, seq_length):
        inputs = ids[:, i:i + seq_length]
        targets = ids[:, (i + 1):(i + 1) + seq_length]

        loss, grads = grad_fn(inputs, states, targets)
        ops.clip_by_global_norm(model.trainable_params())
        optimizer(grads)

        step = (i + 1) // seq_length
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
    # Set intial hidden ane cell states
    state = (ops.zeros((num_layers, 1, hidden_size)),
             ops.zeros((num_layers, 1, hidden_size)))

    # Select one word id randomly
    prob = ops.ones(vocab_size)
    input = ops.multinomial(prob, num_samples=1).unsqueeze(1)

    for i in range(num_samples):
        # Forward propagate RNN
        output, state = model(input, state)

        # Sample a word id
        prob = output.exp()
        word_id = ops.multinomial(prob, num_samples=1).item()

        # Fill input with sampled word id for the next time step
        input.fill_(word_id)

        # File write
        word = corpus.dictionary.idx2word[word_id]
        word = '\n' if word == '<eos>' else word + ' '
        f.write(word)

        if (i + 1) % 100 == 0:
            print('Sampled [{}/{}] words and save to {}'.format(i + 1, num_samples, 'sample.txt'))

# 保存模型
save_path = './lm.ckpt'
mindspore.save_checkpoint(model, save_path)

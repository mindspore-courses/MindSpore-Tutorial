"""数据处理工具"""
import mindspore
from mindspore import ops
from mindspore.common import dtype as mstype


class Dictionary():
    """字典"""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """加入词语"""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class Corpus():
    """语料库"""

    def __init__(self):
        self.dictionary = Dictionary()

    def get_data(self, path, batch_size=20):
        """获取数据"""
        with open(path, 'r', encoding='UTF-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

                    # Tokenize the file content
        ids = ops.zeros(tokens, mstype.int64).asnumpy()
        # ids = mindspore.Tensor(zeros, dtype=mstype.int64)
        token = 0
        with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        ids = mindspore.Tensor(ids, dtype=mstype.int64)
        num_batches = ids.shape[0] // batch_size
        ids = ids[:num_batches * batch_size]
        return ids.view(batch_size, -1)

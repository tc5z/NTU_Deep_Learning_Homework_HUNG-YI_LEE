import torch
from gensim.models import Word2Vec


class PreProcess():
    def __init__(self, sentences, sen_len, w2v_path):
        self.w2v_path = w2v_path  # 模型存储地址
        self.sentences = sentences  # 句子
        self.sen_len = sen_len  # 句子长度
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []  # 词向量矩阵

    def get_w2v_model(self):
        # 读取之前训练好的 word2vec
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        # 这里的 word 只会是 "<PAD>" 或 "<UNK>"
        # 把一个随机生成的表征向量 vector 作为 "<PAD>" 或 "<UNK>" 的嵌入
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.idx2word.append(word)
        self.word2idx[word] = len(self.word2idx)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self, load=True):
        # 生成词向量矩阵
        print("Get embedding ...")
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()  # 获取训练好的 Word2vec word embedding
        else:
            raise NotImplementedError

        for i, word in enumerate(self.embedding.wv.key_to_index):  # 遍历词向量
            print('\r当前构建词向量矩阵进度:{:.2f}%'.format(i / len(self.embedding.wv) * 100), end='')
            self.idx2word.append(word)  # idx2word是一个列表，列表的下标索引对应了单词
            self.word2idx[word] = len(self.word2idx)
            self.embedding_matrix.append(
                self.embedding.wv[word])  # 在embedding_matrix中加入词向量，word所对应的索引就是词向量在embedding_matrix所在的行
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)  # 转成tensor
        # 将 <PAD> 和 <UNK> 加入 embedding
        self.add_embedding("<PAD>")  # 训练时需要将每个句子调整成相同的长度，短的句子需要补<PAD>
        self.add_embedding("<UNK>")  # word2vec时有些词频低的被删掉了，所以有些词可能没有词向量，对于这种词，统一用一个随机的<UNK>词向量表示
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        # 将句子调整成相同长度，即sen_len
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]  # 截断
        else:
            pad_len = self.sen_len - len(sentence)  # 补<PAD>
            for _ in range(pad_len):
                sentence.append(self.word2idx['<PAD>'])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        # 将句子单词用词向量索引表示
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            sentence_idx = []
            for word in sen:
                if word in self.word2idx.keys():
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])  # 表中没有的词用<UNK>表示
            sentence_idx = self.pad_sequence(sentence_idx)  # 调整长度
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)  # torch.size(句子数, sen_len)

    def labels_to_tensor(self, y):
        # 把 labels 转成 tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)

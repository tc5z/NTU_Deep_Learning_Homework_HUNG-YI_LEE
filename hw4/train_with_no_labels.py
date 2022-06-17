from sklearn.model_selection import train_test_split
from utils import load_training_data, load_testing_data
from preprocess import PreProcess
from model import LSTM_Net
from data import TwitterDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from gensim.models import Word2Vec


# 数据预处理
class Preprocess():
    def __init__(self, sen_len, w2v_path):
        self.w2v_path = w2v_path  # word2vec的存储路径
        self.sen_len = sen_len  # 句子的固定长度
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    def get_w2v_model(self):
        # 读取之前训练好的 word2vec
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        # 这里的 word 只会是 "<PAD>" 或 "<UNK>"
        # 把一个随机生成的表征向量 vector 作为 "<PAD>" 或 "<UNK>" 的嵌入
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        # 它的 index 是 word2idx 这个词典的长度，即最后一个
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self, load=True):
        print("Get embedding ...")
        # 获取训练好的 Word2vec word embedding
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # 遍历嵌入后的单词
        for i, word in enumerate(self.embedding.wv.key_to_index):
            print('get words #{}'.format(i + 1), end='\r')
            # 新加入的 word 的 index 是 word2idx 这个词典的长度，即最后一个
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding.wv[word])
        print('')
        # 把 embedding_matrix 变成 tensor
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # 将 <PAD> 和 <UNK> 加入 embedding
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        # 将每个句子变成一样的长度，即 sen_len 的长度
        if len(sentence) > self.sen_len:
            # 如果句子长度大于 sen_len 的长度，就截断
            sentence = sentence[:self.sen_len]
        else:
            # 如果句子长度小于 sen_len 的长度，就补上 <PAD> 符号，缺多少个单词就补多少个 <PAD>
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self, sentences):
        # 把句子里面的字变成相对应的 index
        sentence_list = []
        for i, sen in enumerate(sentences):
            print('sentence count #{}'.format(i + 1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    # 没有出现过的单词就用 <UNK> 表示
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 将每个句子变成一样的长度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        # 把 labels 转成 tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)

    def get_pad(self):
        return self.word2idx["<PAD>"]


def evaluation(outputs, labels):
    # outputs => 预测值，概率（float）
    # labels => 真实值，标签（0或1）
    outputs[outputs >= 0.5] = 1  # 大于等于0.5为正面
    outputs[outputs < 0.5] = 0  # 小于0.5为负面
    accuracy = torch.sum(torch.eq(outputs, labels)).item()
    return accuracy


def add_label(outputs, threshold=0.9):
    id = (outputs >= threshold) | (outputs < 1 - threshold)
    outputs[outputs >= threshold] = 1  # 大于等于 threshold 为正面
    outputs[outputs < 1 - threshold] = 0  # 小于 threshold 为负面
    return outputs.long(), id


def training(batch_size, n_epoch, lr, X_train, y_train, valid, train_x_no_label, model, device):
    # 输出模型总的参数数量、可训练的参数数量
    total = sum(p.numel() for p in model.parameters())  # 返回数组中元素的个数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))

    loss = nn.BCELoss()  # 定义损失函数为二元交叉熵损失 binary cross entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer用Adam，设置适当的学习率l
    v_batch = len(valid)
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):

        train_dataset = TwitterDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        t_batch = len(train_loader)
        total_loss, total_acc = 0, 0

        # training
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, dtype=torch.long)  # 因为 device 为 "cuda"，将 inputs 转成 torch.cuda.LongTensor
            labels = labels.to(device,
                               dtype=torch.float)  # 因为 device 为 "cuda"，将 labels 转成 torch.cuda.FloatTensor，loss()需要float

            optimizer.zero_grad()  # 由于 loss.backward() 的 gradient 会累加，所以每一个 batch 后需要归零
            outputs = model(inputs)  # 模型输入Input，输出output
            outputs = outputs.squeeze()  # 去掉最外面的 dimension，好让 outputs 可以丢进 loss()
            batch_loss = loss(outputs, labels)  # 计算模型此时的 training loss
            batch_loss.backward()  # 计算 loss 的 gradient
            optimizer.step()  # 更新模型参数

            accuracy = evaluation(outputs, labels)  # 计算模型此时的 training accuracy
            total_acc += (accuracy / batch_size)
            total_loss += batch_loss.item()
        print('Epoch | {}/{}'.format(epoch + 1, n_epoch))
        print('Train | Loss:{:.5f} Acc: {:.3f}'.format(total_loss / t_batch, total_acc / t_batch * 100))

        model.eval()
        # self-training
        if epoch > 5:
            train_no_label_dataset = TwitterDataset(X=train_x_no_label, y=None)
            train_no_label_loader = DataLoader(train_no_label_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=0)
            with torch.no_grad():
                for i, (inputs) in enumerate(train_no_label_loader):
                    inputs = inputs.to(device, dtype=torch.long)  # 因为 device 为 "cuda"，将 inputs 转成 torch.cuda.LongTensor

                    outputs = model(inputs)  # 模型输入Input，输出output
                    outputs = outputs.squeeze()
                    labels, id = add_label(outputs)
                    # 加入新标注的数据
                    X_train = torch.cat((X_train.to(device), inputs[id]), dim=0)
                    y_train = torch.cat((y_train.to(device), labels[id]), dim=0)
                    if i == 0:
                        train_x_no_label = inputs[~id]
                    else:
                        train_x_no_label = torch.cat((train_x_no_label.to(device), inputs[~id]), dim=0)

        # validation
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0

            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.float)

                outputs = model(inputs)
                outputs = outputs.squeeze()
                batch_loss = loss(outputs, labels)
                accuracy = evaluation(outputs, labels)
                total_acc += (accuracy / batch_size)
                total_loss += batch_loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss / v_batch, total_acc / v_batch * 100))
            '''
            if total_acc > best_acc:
                # 如果 validation 的结果优于之前所有的結果，就把当下的模型保存下来，用于之后的testing
                best_acc = total_acc
                torch.save(model, "ckpt.model")
            '''
        print('-----------------------------------------------')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sen_len = 20
fix_embedding = True
batch_size = 128
epoch = 15
lr = 0.001
# w2v_path = 'w2v.model'
w2v_path = 'w2v_new.model'

print('loading data...')
train_x, train_y = load_training_data()
train_x_no_label = load_training_data('./training_nolabel.txt')

preprocess = Preprocess(sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)

train_x = preprocess.sentence_word2idx(train_x)
train_y = preprocess.labels_to_tensor(train_y)

train_x_no_label = preprocess.sentence_word2idx(train_x_no_label)

model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
model = model.to(device)
# stratify=y是指按照y标签来分层，也就是数据分层后标签的比例大致等同于原先标签比例
X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.1, random_state=1, stratify=train_y)

val_dataset = TwitterDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 开始训练
training(batch_size, epoch, lr, train_x, train_y, val_loader, train_x_no_label, model, device)

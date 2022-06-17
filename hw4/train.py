from sklearn.model_selection import train_test_split
from utils import load_training_data, load_testing_data
from preprocess import PreProcess
from model import LSTM_Net
from data import TwitterDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pandas as pd


def evaluation(outputs, labels):
    # outputs => 预测值，概率（float）
    # labels => 真实值，标签（0或1）
    outputs[outputs >= 0.5] = 1  # 大于等于0.5为正面
    outputs[outputs < 0.5] = 0  # 小于0.5为负面
    accuracy = torch.sum(torch.eq(outputs, labels)).item()
    return accuracy


def training(batch_size, n_epoch, lr, train, valid, model, device):
    # 输出模型总的参数数量、可训练的参数数量
    total = sum(p.numel() for p in model.parameters())  # 返回数组中元素的个数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))

    loss = nn.BCELoss()  # 定义损失函数为二元交叉熵损失 binary cross entropy loss
    t_batch = len(train)  # training数据的batch数量
    v_batch = len(valid)  # validation数据的batch数量
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # optimizer用Adam，设置适当的学习率lr
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0

        # training
        model.train()
        for i, (inputs, labels) in enumerate(train):
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
            if total_acc > best_acc:
                # 如果 validation 的结果优于之前所有的結果，就把当下的模型保存下来，用于之后的testing
                best_acc = total_acc
                torch.save(model, "ckpt.model")
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

pre_process = PreProcess(train_x, sen_len, w2v_path)
embedding = pre_process.make_embedding(load=True)
train_x = pre_process.sentence_word2idx()
train_y = pre_process.labels_to_tensor(train_y)

model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
model = model.to(device)
# stratify=y是指按照y标签来分层，也就是数据分层后标签的比例大致等同于原先标签比例
X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.1, random_state=1, stratify=train_y)

train_dataset = TwitterDataset(X_train, y_train)
val_dataset = TwitterDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 开始训练
training(batch_size, epoch, lr, train_loader, val_loader, model, device)

'''
def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs >= 0.5] = 1  # 大於等於 0.5 為正面
            outputs[outputs < 0.5] = 0  # 小於 0.5 為負面
            ret_output += outputs.int().tolist()

    return ret_output


test_x = load_testing_data('testing_data.txt')
preprocess = PreProcess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()
test_dataset = TwitterDataset(X=test_x, y=None)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
print('\nload model ...')
outputs = testing(batch_size, test_loader, model, device)

# 寫到 csv 檔案供上傳 Kaggle
tmp = pd.DataFrame({"id": [str(i) for i in range(len(test_x))], "label": outputs})
print("save csv ...")
tmp.to_csv('predict.csv', index=False)
print("Finish Predicting")
'''

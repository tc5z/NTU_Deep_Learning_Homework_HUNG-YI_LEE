import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import glob
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
from PIL import Image


def ConvBlock(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU(),
                         nn.MaxPool2d(kernel_size=2, stride=2))  # 原作者在 paper 裡是說她在 omniglot 用的是 strided convolution


def ConvBlockFunction(x, w, b, w_bn, b_bn):
    x = F.conv2d(x, w, b, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=w_bn, bias=b_bn, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x


class Classifier(nn.Module):
    def __init__(self, in_ch, k_way):
        super(Classifier, self).__init__()
        self.conv1 = ConvBlock(in_ch, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = ConvBlock(64, 64)
        self.logits = nn.Linear(64, k_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.logits(x)
        return x

    def functional_forward(self, x, params):
        '''
        Arguments:
        x: input images [batch, 1, 28, 28]
        params: 模型的參數，也就是 convolution 的 weight 跟 bias，以及 batchnormalization 的  weight 跟 bias
                這是一個 OrderedDict
        '''
        for block in [1, 2, 3, 4]:
            x = ConvBlockFunction(x, params[f'conv{block}.0.weight'], params[f'conv{block}.0.bias'],
                                  params.get(f'conv{block}.1.weight'), params.get(f'conv{block}.1.bias'))
        x = x.view(x.shape[0], -1)
        x = F.linear(x, params['logits.weight'], params['logits.bias'])
        return x


def create_label(n_way, k_shot):
    return torch.arange(n_way).repeat_interleave(k_shot).long()


def MAML(model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_step=1, inner_lr=0.4, train=True):
    """
    Args:
    x is the input omniglot images for a meta_step, shape = [meta_batch_size, n_way * (k_shot + q_query), 1, 28, 28]
    n_way: 每個分類的 task 要有幾個 class
    k_shot: 每個類別在 training 的時候會有多少張照片
    q_query: 在 testing 時，每個類別會用多少張照片 update
    """
    criterion = loss_fn
    task_loss = []  # 這裡面之後會放入每個 task 的 loss
    task_acc = []  # 這裡面之後會放入每個 task 的 loss
    for meta_batch in x:
        train_set = meta_batch[:n_way * k_shot]  # train_set 是我們拿來 update inner loop 參數的 data
        val_set = meta_batch[n_way * k_shot:]  # val_set 是我們拿來 update outer loop 參數的 data

        fast_weights = OrderedDict(model.named_parameters())
        # 在 inner loop update 參數時，我們不能動到實際參數，因此用 fast_weights 來儲存新的參數 θ'

        for inner_step in range(inner_train_step):  # 這個 for loop 是 Algorithm2 的 line 7~8
            # 實際上我們 inner loop 只有 update 一次 gradients，不過某些 task 可能會需要多次 update inner loop 的 θ'，
            # 所以我們還是用 for loop 來寫
            train_label = create_label(n_way, k_shot).cuda()
            logits = model.functional_forward(train_set, fast_weights)
            loss = criterion(logits, train_label)
            grads = torch.autograd.grad(loss, fast_weights.values(),
                                        create_graph=True)  # 這裡是要計算出 loss 對 θ 的微分 (∇loss)
            fast_weights = OrderedDict((name, param - inner_lr * grad)
                                       for ((name, param), grad) in
                                       zip(fast_weights.items(), grads))  # 這裡是用剛剛算出的 ∇loss 來 update θ 變成 θ'

        val_label = create_label(n_way, q_query).cuda()
        logits = model.functional_forward(val_set, fast_weights)  # 這裡用 val_set 和 θ' 算 logit
        loss = criterion(logits, val_label)  # 這裡用 val_set 和 θ' 算 loss
        task_loss.append(loss)  # 把這個 task 的 loss 丟進 task_loss 裡面
        acc = np.asarray([torch.argmax(logits, -1).cpu().numpy() == val_label.cpu().numpy()]).mean()  # 算 accuracy
        task_acc.append(acc)

    model.train()
    optimizer.zero_grad()
    meta_batch_loss = torch.stack(task_loss).mean()  # 我們要用一整個 batch 的 loss 來 update θ (不是 θ')
    if train:
        meta_batch_loss.backward()
        optimizer.step()
    task_acc = np.mean(task_acc)
    return meta_batch_loss, task_acc


class Omniglot(Dataset):
    def __init__(self, data_dir, k_way, q_query):
        self.file_list = [f for f in glob.glob(data_dir + "**/character*", recursive=True)]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.n = k_way + q_query

    def __getitem__(self, idx):
        sample = np.arange(20)
        np.random.shuffle(sample)  # 這裡是為了等一下要 random sample 出我們要的 character
        img_path = self.file_list[idx]
        img_list = [f for f in glob.glob(img_path + "**/*.png", recursive=True)]
        img_list.sort()
        imgs = [self.transform(Image.open(img_file)) for img_file in img_list]
        imgs = torch.stack(imgs)[sample[:self.n]]  # 每個 character，取出 k_shot + q_query 個
        return imgs

    def __len__(self):
        return len(self.file_list)


n_way = 5
k_shot = 1
q_query = 1
inner_train_step = 1
inner_lr = 0.4
meta_lr = 0.001
meta_batch_size = 32
max_epoch = 40
eval_batches = test_batches = 20
train_data_path = './Omniglot/images_background/'
test_data_path = './Omniglot/images_evaluation/'

#dataset = Omniglot(train_data_path, k_shot, q_query)
train_set, val_set = torch.utils.data.random_split(Omniglot(train_data_path, k_shot, q_query), [3200, 656])
train_loader = DataLoader(train_set,
                          batch_size=n_way,  # 這裡的 batch size 並不是 meta batch size, 而是一個task裡面會有多少不同的
                                             # characters，也就是 few-shot classifiecation 的 n_way
                          shuffle=True,
                          drop_last=True)
val_loader = DataLoader(val_set,
                        batch_size=n_way,
                        shuffle=True,
                        drop_last=True)
test_loader = DataLoader(Omniglot(test_data_path, k_shot, q_query),
                         batch_size=n_way,
                         shuffle=True,
                         drop_last=True)
train_iter = iter(train_loader)
val_iter = iter(val_loader)
test_iter = iter(test_loader)

meta_model = Classifier(1, n_way).cuda()
optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)
loss_fn = nn.CrossEntropyLoss().cuda()


def get_meta_batch(meta_batch_size, k_shot, q_query, data_loader, iterator):
    data = []
    for _ in range(meta_batch_size):
        try:
            task_data = iterator.next()  # 一筆 task_data 就是一個 task 裡面的 data，大小是 [n_way, k_shot+q_query, 1, 28, 28]
        except StopIteration:
            iterator = iter(data_loader)
            task_data = iterator.next()
        train_data = task_data[:, :k_shot].reshape(-1, 1, 28, 28)
        val_data = task_data[:, k_shot:].reshape(-1, 1, 28, 28)
        task_data = torch.cat((train_data, val_data), 0)
        data.append(task_data)
    return torch.stack(data).cuda(), iterator


for epoch in range(max_epoch):
    print("Epoch %d" % epoch)
    train_meta_loss = []
    train_acc = []
    for step in tqdm(range(len(train_loader) // meta_batch_size)):  # 這裡的 step 是一次 meta-gradient update step
        x, train_iter = get_meta_batch(meta_batch_size, k_shot, q_query, train_loader, train_iter)
        meta_loss, acc = MAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn)
        train_meta_loss.append(meta_loss.item())
        train_acc.append(acc)
    print("  Loss    : ", np.mean(train_meta_loss))
    print("  Accuracy: ", np.mean(train_acc))

    # 每個 epoch 結束後，看看 validation accuracy 如何
    # 助教並沒有做 early stopping，同學如果覺得有需要是可以做的
    val_acc = []
    for eval_step in tqdm(range(len(val_loader) // eval_batches)):
        x, val_iter = get_meta_batch(eval_batches, k_shot, q_query, val_loader, val_iter)
        _, acc = MAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_step=3,
                      train=False)  # testing時，我們更新三次 inner-step
        val_acc.append(acc)
    print("  Validation accuracy: ", np.mean(val_acc))

test_acc = []
for test_step in tqdm(range(len(test_loader) // test_batches)):
    x, test_iter = get_meta_batch(test_batches, k_shot, q_query, test_loader, test_iter)
    _, acc = MAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_step=3,
                  train=False)  # testing 時，我們更新三次 inner-step
    test_acc.append(acc)
print("  Testing accuracy: ", np.mean(test_acc))

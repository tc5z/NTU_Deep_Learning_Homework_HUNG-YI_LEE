import torch.nn as nn
import torch.nn.functional as F
import re
import torch
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import torch.optim as optim
import torchvision.models as models
import numpy as np
import pickle


class StudentNet(nn.Module):
    """
        在這個Net裡面，我們會使用Depthwise & Pointwise Convolution Layer來疊model。
        你會發現，將原本的Convolution Layer換成Dw & Pw後，Accuracy通常不會降很多。

        另外，取名為StudentNet是因為這個Model等會要做Knowledge Distillation。

        # 一般的Convolution, weight大小 = in_chs * out_chs * kernel_size^2
        nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding)

        # Group Convolution, Group數目可以自行控制，表示要分成幾群。其中in_chs和out_chs必須要可以被groups整除。(不然沒辦法分群。)
        nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups)

        # Depthwise Convolution, 輸入chs=輸出chs=Groups數目, weight大小 = in_chs * kernel_size^2
        nn.Conv2d(in_chs, out_chs=in_chs, kernel_size, stride, padding, groups=in_chs)

        # Pointwise Convolution, 也就是1 by 1 convolution, weight大小 = in_chs * out_chs
        nn.Conv2d(in_chs, out_chs, 1)
    """

    def __init__(self, base=16, width_mult=1):
        """
            base: 這個model一開始的ch數量，每過一層都會*2，直到base*16為止。
            width_mult: 為了之後的Network Pruning使用，在base*8之后的Layer上會 * width_mult代表剪枝後的ch數量。
        """
        super(StudentNet, self).__init__()
        multiplier = [1, 2, 4, 8, 16, 16, 16, 16]

        # bandwidth: 每一層Layer所使用的ch數量
        bandwidth = [base * m for m in multiplier]

        # 我們只Pruning第三層以後的Layer
        for i in range(3, 7):
            bandwidth[i] = int(bandwidth[i] * width_mult)

        self.cnn = nn.Sequential(
            # 第一層我們通常不會拆解Convolution Layer。
            nn.Sequential(
                nn.Conv2d(3, bandwidth[0], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[0]),
                nn.ReLU6(),
                nn.MaxPool2d(2, 2, 0),
            ),
            # 接下來每一個Sequential Block都一樣，所以我們只講一個Block
            nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[0], 3, 1, 1, groups=bandwidth[0]),
                # Batch Normalization
                nn.BatchNorm2d(bandwidth[0]),
                # ReLU6 是限制Neuron最小只會到0，最大只會到6。 MobileNet系列都是使用ReLU6。
                # 使用ReLU6的原因是因為如果數字太大，會不好壓到float16 / or further qunatization，因此才給個限制。
                nn.ReLU6(),
                # Pointwise Convolution
                nn.Conv2d(bandwidth[0], bandwidth[1], 1),
                # 過完Pointwise Convolution不需要再做ReLU，經驗上Pointwise + ReLU效果都會變差。
                nn.MaxPool2d(2, 2, 0),
                # 每過完一個Block就Down Sampling
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[1], bandwidth[1], 3, 1, 1, groups=bandwidth[1]),
                nn.BatchNorm2d(bandwidth[1]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[1], bandwidth[2], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[2], bandwidth[2], 3, 1, 1, groups=bandwidth[2]),
                nn.BatchNorm2d(bandwidth[2]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[2], bandwidth[3], 1),
                nn.MaxPool2d(2, 2, 0),
            ),

            # 到這邊為止因為圖片已經被Down Sample很多次了，所以就不做MaxPool
            nn.Sequential(
                nn.Conv2d(bandwidth[3], bandwidth[3], 3, 1, 1, groups=bandwidth[3]),
                nn.BatchNorm2d(bandwidth[3]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[3], bandwidth[4], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[4], bandwidth[4], 3, 1, 1, groups=bandwidth[4]),
                nn.BatchNorm2d(bandwidth[4]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[4], bandwidth[5], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[5], bandwidth[5], 3, 1, 1, groups=bandwidth[5]),
                nn.BatchNorm2d(bandwidth[5]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[5], bandwidth[6], 1),
            ),

            nn.Sequential(
                nn.Conv2d(bandwidth[6], bandwidth[6], 3, 1, 1, groups=bandwidth[6]),
                nn.BatchNorm2d(bandwidth[6]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[6], bandwidth[7], 1),
            ),

            # 這邊我們採用Global Average Pooling。
            # 如果輸入圖片大小不一樣的話，就會因為Global Average Pooling壓成一樣的形狀，這樣子接下來做FC就不會對不起來。
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            # 這邊我們直接Project到11維輸出答案。
            nn.Linear(bandwidth[7], 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


def loss_fn_kd(outputs, labels, teacher_outputs, T=20, alpha=0.5):
    # 一般的Cross Entropy
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # 讓logits的log_softmax對目標機率(teacher的logits/T後softmax)做KL Divergence。
    soft_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)
    return hard_loss + soft_loss


class MyDataset(Dataset):
    def __init__(self, folderName, transform=None):
        self.transform = transform
        self.data = []
        self.label = []

        for img_path in sorted(glob(folderName + '/*.jpg')):
            try:
                # Get classIdx by parsing image path
                class_idx = int(re.findall(re.compile(r'\d+'), img_path)[1])
            except:
                # if inference mode (there's no answer), class_idx default 0
                class_idx = 0

            image = Image.open(img_path)
            # Get File Descriptor
            image_fp = image.fp
            image.load()
            # Close File Descriptor (or it'll reach OPEN_MAX)
            image_fp.close()

            self.data.append(image)
            self.label.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.label[idx]


trainTransform = transforms.Compose([
    transforms.RandomCrop(256, pad_if_needed=True, padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
testTransform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])


def get_dataloader(mode='training', batch_size=32):

    assert mode in ['training', 'testing', 'validation']

    dataset = MyDataset(
        f'../food-11/{mode}',
        transform=trainTransform if mode == 'training' else testTransform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'training'))

    return dataloader


# get dataloader
train_dataloader = get_dataloader('training', batch_size=32)
valid_dataloader = get_dataloader('validation', batch_size=32)

'''
teacher_net = models.resnet18(pretrained=False, num_classes=11).cuda()
student_net = StudentNet(base=16).cuda()

teacher_net.load_state_dict(torch.load(f'./teacher_resnet18.bin'))
optimizer = optim.AdamW(student_net.parameters(), lr=1e-3)


def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # 清空 optimizer
        optimizer.zero_grad()
        # 處理 input
        inputs, hard_labels = batch_data
        inputs = inputs.cuda()
        hard_labels = torch.LongTensor(hard_labels).cuda()
        # 因為Teacher沒有要backprop，所以我們使用torch.no_grad
        # 告訴torch不要暫存中間值(去做backprop)以浪費記憶體空間。
        with torch.no_grad():
            soft_labels = teacher_net(inputs)

        if update:
            logits = student_net(inputs)
            # 使用我們之前所寫的融合soft label&hard label的loss。
            # T=20是原始論文的參數設定。
            loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)
            loss.backward()
            optimizer.step()
        else:
            # 只是算validation acc的話，就開no_grad節省空間。
            with torch.no_grad():
                logits = student_net(inputs)
                loss = loss_fn_kd(logits, hard_labels, soft_labels, 20, alpha)

        total_hit += torch.sum(torch.argmax(logits, dim=1) == hard_labels).item()
        total_num += len(inputs)

        total_loss += loss.item() * len(inputs)
    return total_loss / total_num, total_hit / total_num


# TeacherNet永遠都是Eval mode.
teacher_net.eval()
now_best_acc = 0
for epoch in range(200):
    student_net.train()
    train_loss, train_acc = run_epoch(train_dataloader, update=True)
    student_net.eval()
    valid_loss, valid_acc = run_epoch(valid_dataloader, update=False)

    # 存下最好的model。
    if valid_acc > now_best_acc:
        now_best_acc = valid_acc
        torch.save(student_net.state_dict(), 'student_model.bin')

    print('epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
        epoch, train_loss, train_acc, valid_loss, valid_acc))
'''


def network_slimming(old_model, new_model):
    params = old_model.state_dict()
    new_params = new_model.state_dict()

    # selected_idx: 每一層所選擇的neuron index
    selected_idx = []
    # 我們總共有7層CNN，因此逐一抓取選擇的neuron index們。
    for i in range(8):
        # 根據上表，我們要抓的gamma係數在cnn.{i}.1.weight內。
        importance = params[f'cnn.{i}.1.weight']
        # 抓取總共要篩選幾個neuron。
        old_dim = len(importance)
        new_dim = len(new_params[f'cnn.{i}.1.weight'])
        # 以Ranking做Index排序，較大的會在前面(descending=True)。
        ranking = torch.argsort(importance, descending=True)
        # 把篩選結果放入selected_idx中。
        selected_idx.append(ranking[:new_dim])

    now_processed = 1
    for (name, p1), (name2, p2) in zip(params.items(), new_params.items()):
        # 如果是cnn層，則移植參數。
        # 如果是FC層，或是該參數只有一個數字(例如batchnorm的tracenum等等資訊)，那麼就直接複製。
        if name.startswith('cnn') and p1.size() != torch.Size([]) and now_processed != len(selected_idx):
            # 當處理到Pointwise的weight時，讓now_processed+1，表示該層的移植已經完成。
            if name.startswith(f'cnn.{now_processed}.3'):
                now_processed += 1

            # 如果是pointwise，weight會被上一層的pruning和下一層的pruning所影響，因此需要特判。
            if name.endswith('3.weight'):
                # 如果是最後一層cnn，則輸出的neuron不需要prune掉。
                if len(selected_idx) == now_processed:
                    new_params[name] = p1[:, selected_idx[now_processed - 1]]
                # 反之，就依照上層和下層所選擇的index進行移植。
                # 這裡需要注意的是Conv2d(x,y,1)的weight shape是(y,x,1,1)，順序是反的。
                else:
                    new_params[name] = p1[selected_idx[now_processed]][:, selected_idx[now_processed - 1]]
            else:
                new_params[name] = p1[selected_idx[now_processed]]
        else:
            new_params[name] = p1

    # 讓新model load進被我們篩選過的parameters，並回傳new_model。
    new_model.load_state_dict(new_params)
    return new_model


net = StudentNet().cuda()
net.load_state_dict(torch.load('student_custom_small.bin'))

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=1e-3)


def run_epoch(dataloader, update=True, alpha=0.5):
    total_num, total_hit, total_loss = 0, 0, 0
    for now_step, batch_data in enumerate(dataloader):
        # 清空 optimizer
        optimizer.zero_grad()
        # 處理 input
        inputs, labels = batch_data
        inputs = inputs.cuda()
        labels = labels.cuda()

        logits = net(inputs)
        loss = criterion(logits, labels)
        if update:
            loss.backward()
            optimizer.step()

        total_hit += torch.sum(torch.argmax(logits, dim=1) == labels).item()
        total_num += len(inputs)
        total_loss += loss.item() * len(inputs)

    return total_loss / total_num, total_hit / total_num


now_width_mult = 1
for i in range(1):
    now_width_mult *= 0.95
    new_net = StudentNet(width_mult=now_width_mult).cuda()
    params = net.state_dict()
    net = network_slimming(net, new_net)
    now_best_acc = 0
    for epoch in range(5):
        net.train()
        train_loss, train_acc = run_epoch(train_dataloader, update=True)
        net.eval()
        valid_loss, valid_acc = run_epoch(valid_dataloader, update=False)
        # 在每個width_mult的情況下，存下最好的model。
        if valid_acc > now_best_acc:
            now_best_acc = valid_acc
            torch.save(net.state_dict(), f'custom_small_rate_{now_width_mult}.bin')
        print('rate {:6.4f} epoch {:>3d}: train loss: {:6.4f}, acc {:6.4f} valid loss: {:6.4f}, acc {:6.4f}'.format(
            now_width_mult,
            epoch, train_loss, train_acc, valid_loss, valid_acc))


def encode16(params, fname):
    '''將params壓縮成16-bit後輸出到fname。

    Args:
      params: model的state_dict。
      fname: 壓縮後輸出的檔名。
    '''

    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        # 有些東西不屬於ndarray，只是一個數字，這個時候我們就不用壓縮。
        if type(param) == np.ndarray:
            custom_dict[name] = np.float16(param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))


def decode16(fname):
    '''從fname讀取各個params，將其從16-bit還原回torch.tensor後存進state_dict內。

    Args:
      fname: 壓縮後的檔名。
    '''

    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        param = torch.tensor(param)
        custom_dict[name] = param

    return custom_dict


# encode16(params, '16_bit_model.pkl')
# print(f"16-bit cost: {os.stat('16_bit_model.pkl').st_size} bytes.")

def encode8(params, fname):
    custom_dict = {}
    for (name, param) in params.items():
        param = np.float64(param.cpu().numpy())
        if type(param) == np.ndarray:
            min_val = np.min(param)
            max_val = np.max(param)
            param = np.round((param - min_val) / (max_val - min_val) * 255)
            param = np.uint8(param)
            custom_dict[name] = (min_val, max_val, param)
        else:
            custom_dict[name] = param

    pickle.dump(custom_dict, open(fname, 'wb'))


def decode8(fname):
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        if type(param) == tuple:
            min_val, max_val, param = param
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)

        custom_dict[name] = param

    return custom_dict

# encode8(params, '8_bit_model.pkl')
# print(f"8-bit cost: {os.stat('8_bit_model.pkl').st_size} bytes.")

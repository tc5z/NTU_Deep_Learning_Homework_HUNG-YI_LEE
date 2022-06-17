import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

device = 'cpu'


def meta_task_data(seed=0, a_range=[0.1, 5], b_range=[0, 2 * np.pi], task_num=100,
                   n_sample=10, sample_range=[-5, 5], plot=False):
    np.random.seed(seed)
    a_s = np.random.uniform(low=a_range[0], high=a_range[1], size=task_num)
    b_s = np.random.uniform(low=b_range[0], high=b_range[1], size=task_num)
    total_x = []
    total_y = []
    label = []
    for t in range(task_num):
        x = np.random.uniform(low=sample_range[0], high=sample_range[1], size=n_sample)
        total_x.append(x)
        total_y.append(a_s[t] * np.sin(x + b_s[t]))
        label.append('{:.3}*sin(x+{:.3})'.format(a_s[t], b_s[t]))
    if plot:
        plot_x = [np.linspace(-5, 5, 1000)]
        plot_y = []
        for t in range(task_num):
            plot_y.append(a_s[t] * np.sin(plot_x + b_s[t]))
        return total_x, total_y, plot_x, plot_y, label
    else:
        return total_x, total_y, label


class MetaLinear(nn.Module):
    def __init__(self, init_layer=None):
        super(MetaLinear, self).__init__()
        if type(init_layer) != type(None):
            self.weight = init_layer.weight.clone()
            self.bias = init_layer.bias.clone()

    def zero_grad(self):
        self.weight.grad = torch.zeros_like(self.weight)
        self.bias.grad = torch.zeros_like(self.bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class net(nn.Module):
    def __init__(self, init_weight=None):
        super(net, self).__init__()
        if type(init_weight) != type(None):
            for name, module in init_weight.named_modules():
                if name != '':
                    setattr(self, name, MetaLinear(module))
        else:
            self.hidden1 = nn.Linear(1, 40)
            self.hidden2 = nn.Linear(40, 40)
            self.out = nn.Linear(40, 1)

    def update(self, parent, lr=1):
        layers = self.__dict__['_modules']
        parent_layers = parent.__dict__['_modules']
        for param in layers.keys():
            layers[param].weight = layers[param].weight - lr * parent_layers[param].weight.grad
            layers[param].bias = layers[param].bias - lr * parent_layers[param].bias.grad
        # gradient will flow back due to clone backward

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.out(x)


class Meta_learning_model():
    def __init__(self, init_weight=None):
        super(Meta_learning_model, self).__init__()
        self.model = net().to(device)
        if type(init_weight) != type(None):
            self.model.load_state_dict(init_weight)
        self.grad_buffer = 0

    def gen_models(self, num, check=True):
        models = [net(init_weight=self.model).to(device) for i in range(num)]
        return models

    def clear_buffer(self):
        print("Before grad", self.grad_buffer)
        self.grad_buffer = 0


bsz = 10
train_x, train_y, train_label = meta_task_data(task_num=50000 * 10)
train_x = torch.Tensor(train_x).unsqueeze(-1)  # add one dim
train_y = torch.Tensor(train_y).unsqueeze(-1)
train_dataset = data.TensorDataset(train_x, train_y)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=bsz, shuffle=False)

test_x, test_y, plot_x, plot_y, test_label = meta_task_data(task_num=1, n_sample=10, plot=True)
test_x = torch.Tensor(test_x).unsqueeze(-1)  # add one dim
test_y = torch.Tensor(test_y).unsqueeze(-1)  # add one dim
plot_x = torch.Tensor(plot_x).unsqueeze(-1)  # add one dim
test_dataset = data.TensorDataset(test_x, test_y)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=bsz, shuffle=False)

meta_model = Meta_learning_model()
meta_optimizer = torch.optim.Adam(meta_model.model.parameters(), lr=1e-3)

pretrain = net()
pretrain.to(device)
pretrain.train()
pretrain_optim = torch.optim.Adam(pretrain.parameters(), lr=1e-3)

epoch = 1
for e in range(epoch):
    meta_model.model.train()
    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)
        sub_models = meta_model.gen_models(bsz)

        meta_l = 0
        for model_num in range(len(sub_models)):
            sample = list(range(10))
            np.random.shuffle(sample)

            # pretraining
            pretrain_optim.zero_grad()
            y_tilde = pretrain(x[model_num][sample[:5], :])
            little_l = F.mse_loss(y_tilde, y[model_num][sample[:5], :])
            little_l.backward()
            pretrain_optim.step()
            pretrain_optim.zero_grad()
            y_tilde = pretrain(x[model_num][sample[5:], :])
            little_l = F.mse_loss(y_tilde, y[model_num][sample[5:], :])
            little_l.backward()
            pretrain_optim.step()

            # meta learning
            y_tilde = sub_models[model_num](x[model_num][sample[:5], :])
            little_l = F.mse_loss(y_tilde, y[model_num][sample[:5], :])
            # 計算第一次 gradient 並保留計算圖以接著計算更高階的 gradient
            little_l.backward(create_graph=True)
            sub_models[model_num].update(lr=1e-2, parent=meta_model.model)
            # 先清空 optimizer 中計算的 gradient 值 (避免累加)
            meta_optimizer.zero_grad()

            # 計算第二次 (二階) 的 gradient，二階的原因來自第一次 update 時有計算過一次 gradient 了
            y_tilde = sub_models[model_num](x[model_num][sample[5:], :])
            meta_l = meta_l + F.mse_loss(y_tilde, y[model_num][sample[5:], :])

        meta_l = meta_l / bsz
        meta_l.backward()
        meta_optimizer.step()
        meta_optimizer.zero_grad()

test_model = copy.deepcopy(meta_model.model)
test_model.train()
test_optim = torch.optim.SGD(test_model.parameters(), lr=1e-3)

fig = plt.figure(figsize=[9.6, 7.2])
ax = plt.subplot(111)
plot_x1 = plot_x.squeeze().numpy()
ax.scatter(test_x.numpy().squeeze(), test_y.numpy().squeeze())
ax.plot(plot_x1, plot_y[0].squeeze())

test_model.train()
pretrain.train()

for epoch in range(10):
    for x, y in test_loader:
        y_tilde = test_model(x[0])
        little_l = F.mse_loss(y_tilde, y[0])
        test_optim.zero_grad()
        little_l.backward()
        test_optim.step()
        print("(meta)))Loss: ", little_l.item())

for epoch in range(10):
    for x, y in test_loader:
        y_tilde = pretrain(x[0])
        little_l = F.mse_loss(y_tilde, y[0])
        pretrain_optim.zero_grad()
        little_l.backward()
        pretrain_optim.step()
        print("(pretrain)Loss: ", little_l.item())

test_model.eval()
pretrain.eval()

plot_y_tilde = test_model(plot_x[0]).squeeze().detach().numpy()
plot_x2 = plot_x.squeeze().numpy()
ax.plot(plot_x2, plot_y_tilde, label='tune(disjoint)')
ax.legend()
fig.show()

plot_y_tilde = pretrain(plot_x[0]).squeeze().detach().numpy()
plot_x2 = plot_x.squeeze().numpy()
ax.plot(plot_x2, plot_y_tilde, label='pretrain')
ax.legend()
fig.show()

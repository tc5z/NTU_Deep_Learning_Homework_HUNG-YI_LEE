import os
from torch.utils.data import DataLoader
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image
from pdb import set_trace
import time


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        def building_block(indim, outdim):
            return [
                nn.Conv2d(indim, outdim, 3, 1, 1),
                nn.BatchNorm2d(outdim),
                nn.ReLU(),
            ]

        def stack_blocks(indim, outdim, block_num):
            layers = building_block(indim, outdim)
            for i in range(block_num - 1):
                layers += building_block(outdim, outdim)
            layers.append(nn.MaxPool2d(2, 2, 0))
            return layers

        cnn_list = []
        cnn_list += stack_blocks(3, 128, 1)
        cnn_list += stack_blocks(128, 128, 1)
        cnn_list += stack_blocks(128, 256, 1)
        cnn_list += stack_blocks(256, 512, 1)
        cnn_list += stack_blocks(512, 512, 1)
        self.cnn = nn.Sequential(*cnn_list)

        dnn_list = [
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11),
        ]
        self.fc = nn.Sequential(*dnn_list)

    def forward(self, x):
        out = self.cnn(x)
        out = out.reshape(out.size()[0], -1)
        return self.fc(out)


class FoodDataset(Dataset):
    def __init__(self, paths, labels, mode):
        # mode: 'train' or 'eval'

        self.paths = paths
        self.labels = labels
        trainTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        evalTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
        ])
        self.transform = trainTransform if mode == 'train' else evalTransform

    # ?????? FoodDataset ????????? pytorch ??? Dataset class
    # ??? __len__ ??? __getitem__ ??????????????? pytorch dataset ???????????? implement ????????? methods
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        X = Image.open(self.paths[index])
        X = self.transform(X)
        Y = self.labels[index]
        return X, Y

    # ?????? method ????????? pytorch dataset ??????????????????????????????????????????????????????????????????????????????????????? batch ??? visualize
    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
            image, label = self.__getitem__(index)
            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)


# ?????? data ??????????????????????????????????????????????????????class???
def get_paths_labels(path):
    imgnames = os.listdir(path)
    imgnames.sort()
    imgpaths = []
    labels = []
    for name in imgnames:
        imgpaths.append(os.path.join(path, name))
        labels.append(int(name.split('_')[0]))
    return imgpaths, labels


train_paths, train_labels = get_paths_labels('D:\\360Downloads\\cs_data\\food-11\\training')
# ????????? initialize dataset ???????????????????????????class?????????????????? dataset ????????????
# dataset ??? __getitem__ method ?????????????????? load ???????????????????????????
train_set = FoodDataset(train_paths, train_labels, mode='train')
val_paths, val_labels = get_paths_labels('D:\\360Downloads\\cs_data\\food-11\\validation')
val_set = FoodDataset(val_paths, val_labels, mode='eval')

batch_size = 128

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Classifier().to(device)
loss = nn.CrossEntropyLoss()  # ?????????????????????
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer ?????? Adam
num_epoch = 15  # ????????????

# ??????
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0  # ????????????epoch??????????????????
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()  # ?????? model ?????? train model ????????? Dropout ???...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()  # ??? optimizer ??? model ????????? gradient ??????
        train_pred = model(data[0].to(device))  # ?????? model ????????????????????????????????????
        batch_loss = loss(train_pred, data[1].to(device))  # ?????? loss ????????? prediction ??? label ??????????????? CPU ?????? GPU ??????
        batch_loss.backward()  # ?????? back propagation ????????????????????? gradient
        optimizer.step()  # ??? optimizer ??? gradient ???????????????
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].to(device))
            batch_loss = loss(val_pred, data[1].to(device))

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        # ????????? print ??????
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' %
              (epoch + 1, num_epoch, time.time() - epoch_start_time, train_acc / len(train_set),
               train_loss / len(train_set), val_acc / len(val_set), val_loss / len(val_set)))


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def compute_saliency_maps(x, y, model):
    model.eval()
    x = x.cuda()

    # ?????????????????? code
    # ????????????????????? loss ??? input image ?????????????????? input x ???????????? tensor?????????????????? gradient
    # ??????????????????????????? pytorch ?????? input x ??????gradient????????????????????? backward ??? x.grad ?????????????????????
    x.requires_grad_()

    y_pred = model(x)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()

    saliencies = x.grad.abs().detach().cpu()
    # saliencies: (batches, channels, height, weight)
    # ?????????????????????????????????????????? saliency map?????????????????? gradient scale ????????????????????????
    # ???????????????????????? gradient ??? 100 ~ 1000???????????????????????? gradient ??? 0.001 ~ 0.0001
    # ????????????????????????????????????????????? saliency ???????????????????????????????????????????????????????????????????????????
    # ????????????????????????????????????????????????????????????????????? saliency ???????????????????????????
    # ?????????????????????????????? saliency ????????? normalize???????????????????????????????????????????????????
    saliencies = torch.stack([normalize(item) for item in saliencies])
    return saliencies


# ?????????????????? visualize ????????? indices
img_indices = [83, 4218, 4707, 8598]
images, labels = train_set.getbatch(img_indices)
saliencies = compute_saliency_maps(images, labels, model)

# ?????? matplotlib ?????????
fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for row, target in enumerate([images, saliencies]):
  for column, img in enumerate(target):
    axs[row][column].imshow(img.permute(1, 2, 0).numpy())
    # ????????????permute ??????????????????????????????????
    # ??? pytorch ????????????image tensor ??? dimension ?????????????????? (channels, height, width)
    # ?????? matplolib ??????????????????????????? tensor ??????????????????????????? (height, width, channels)
    # ?????? permute ????????? pytorch ???????????????????????? dimension ????????????
    # ?????? img.permute(1, 2, 0)????????????????????? tensor??????
    # - ??? 0 ??? dimension ????????? img ?????? 1 ??? dimension???????????? height
    # - ??? 1 ??? dimension ????????? img ?????? 2 ??? dimension???????????? width
    # - ??? 2 ??? dimension ????????? img ?????? 0 ??? dimension???????????? channels
plt.show()
plt.close()

layer_activations = None


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def filter_explaination(x, model, cnnid, filterid, iteration=100, lr=1):
    # x: ????????????????????????????????? activate ????????? filter ????????????
    # cnnid, filterid: ????????????????????? cnn ???????????? filter
    model.eval()

    def hook(model, input, output):
        global layer_activations
        layer_activations = output

    hook_handle = model.cnn[cnnid].register_forward_hook(hook)
    # ????????????????????? pytorch?????? forward ??????????????? cnnid ??? cnn ?????????????????? hook ????????????????????? function ?????????????????? forward ????????? cnn
    # ??????????????? hook function ?????????????????????????????? output???????????? activation map ????????????????????? forward ????????? model ????????????????????? loss
    # ???????????? cnn ??? activation map
    # ??????????????????????????????????????????????????? forward???????????????????????? pytorch ??????????????? forward ?????????????????????
    # ?????????hook_handle ????????????????????????????????????????????????????????????

    # Filter activation: ??????????????? x ??????????????? filter ??? activation map
    model(x.cuda())
    # ???????????????????????? forward???????????????????????? activation map??????????????????????????? loss ?????????
    filter_activations = layer_activations[:, filterid, :, :].detach().cpu()

    # ?????? function argument ????????? filterid ????????? filter ??? activation map ?????????
    # ?????????????????? activation map ??????????????????????????????????????????????????? detach from graph ????????? cpu tensor

    # Filter visualization: ??????????????????????????????????????? activate ??? filter ?????????
    x = x.cuda()
    # ????????? random noise ?????????????????? (?????????????????? dataset image ?????????)
    x.requires_grad_()
    # ???????????? input image ????????????
    optimizer = Adam([x], lr=lr)
    # ?????????????????? optimizer??????????????? input image ?????? filter activation ????????????
    for iter in range(iteration):
        optimizer.zero_grad()
        model(x)

        objective = -layer_activations[:, filterid, :, :].sum()
        # ?????????????????????????????????????????????????????? image ?????????????????????????????? final loss
        # ????????????????????????image ?????????????????????????????? activation ?????????
        # ?????? objective ??? filter activation ???????????????????????????????????????????????? maximization

        objective.backward()
        # ?????? filter activation ??? input image ????????????
        optimizer.step()
        # ?????? input image ???????????? filter activation
    filter_visualization = x.detach().cpu().squeeze()[0]
    # ??????????????????????????????????????????????????????????????? detach ????????? cpu tensor

    hook_handle.remove()
    # ????????????????????? model register hook?????? hook ???????????????????????????????????? register ?????? hook
    # ??? model ?????? forward ???????????????????????????????????????????????????????????????????????? (???????????????????????????????????? hook ???)
    # ?????????????????????????????????????????? hook ???????????????????????????????????? register ????????????

    return filter_activations, filter_visualization


img_indices = [83, 4218, 4707, 8598]
images, labels = train_set.getbatch(img_indices)
filter_activations, filter_visualization = filter_explaination(images, model, cnnid=18, filterid=0, iteration=100, lr=0.1)

# ?????? filter visualization
plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
plt.show()
plt.close()

# ?????? filter activations
fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for i, img in enumerate(images):
  axs[0][i].imshow(img.permute(1, 2, 0))
for i, img in enumerate(filter_activations):
  axs[1][i].imshow(normalize(img))
plt.show()
plt.close()


#%%
def predict(input):
    # input: numpy array, (batches, height, width, channels)

    model.eval()
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)
    # ???????????? input ?????? pytorch tensor???????????? pytorch ????????? dimension ??????
    # ????????? (batches, channels, height, width)

    output = model(input.cuda())
    return output.detach().cpu().numpy()


def segmentation(input):
    # ?????? skimage ????????? segmentation ??????????????? 100 ???
    return slic(input, n_segments=100, compactness=1, sigma=1)


img_indices = [83, 4218, 4707, 8598]
images, labels = train_set.getbatch(img_indices)
fig, axs = plt.subplots(1, 4, figsize=(15, 8))
np.random.seed(16)
# ????????? reproducible
for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
    x = image.astype(np.double)
    # lime ?????????????????? numpy array

    explainer = lime_image.LimeImageExplainer()
    explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)
    # ???????????????????????? lime explainer ??????????????? function?????????????????????
    # classifier_fn ???????????????????????? model ?????? prediction
    # segmentation_fn ???????????????????????? segmentation
    # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=explain_instance#lime.lime_image.LimeImageExplainer.explain_instance

    lime_img, mask = explaination.get_image_and_mask(
        label=label.item(),
        positive_only=False,
        hide_rest=False,
        num_features=11,
        min_weight=0.05
    )
    # ??? explainer ???????????????????????????
    # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=get_image_and_mask#lime.lime_image.ImageExplanation.get_image_and_mask

    axs[idx].imshow(lime_img)

plt.show()
plt.close()

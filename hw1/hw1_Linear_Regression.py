import sys
import math
import pandas as pd
import numpy as np


# 读入train.csv，繁体字以big5编码
data = pd.read_csv('./train.csv', encoding='big5')
# 丢弃前两列，需要的是从第三列开始的数值
data = data.iloc[:, 3:]
# 把降雨的NR字符变成数值0
data[data == 'NR'] = 0
# 把dataframe转换成numpy的数组
raw_data = data.to_numpy()

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample

x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour : day * 24 + hour + 9].reshape(1, -1)  # vector dim:18*9
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  # value


mean_x = np.mean(x, axis=0)  # 18 * 9
std_x = np.std(x, axis=0)  # 18 * 9
for i in range(len(x)):  # 12 * 471
    for j in range(len(x[0])):  # 18 * 9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

dim = 18 * 9 + 1
w = np.zeros([dim, 1])
x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)
learning_rate = 0.1
iter_time = 500
# adagrad = np.zeros([dim, 1])
mt = np.zeros([dim, 1])  # adam
mt_correct = np.zeros([dim, 1])
vt = np.zeros([dim, 1])
vt_correct = np.zeros([dim, 1])
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
for t in range(1, iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)  # rmse
    if t % 100 == 0:
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  # dim*1
    # adagrad += gradient ** 2
    # w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    mt = beta1 * mt + (1 - beta1) * gradient
    vt = beta2 * vt + (1 - beta2) * gradient ** 2
    mt_correct = mt / (1 - beta1 ** t)
    vt_correct = vt / (1 - beta2 ** t)
    w = w - learning_rate * mt_correct / np.sqrt(vt_correct + eps)


# 读入测试数据test.csv
testdata = pd.read_csv('./test.csv', header=None, encoding='big5')
# 丢弃前两列，需要的是从第3列开始的数据
test_data = testdata.iloc[:, 2:]
# 把降雨为NR字符变成数字0
test_data[test_data == 'NR'] = 0
# 将dataframe变成numpy数组
test_data = test_data.to_numpy()
# 将test数据也变成 240 个维度为 18 * 9 + 1 的数据。
test_x = np.empty([240, 18*9], dtype=float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)
ans_y = np.dot(test_x, w)


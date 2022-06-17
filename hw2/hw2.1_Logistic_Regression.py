import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X_train_fpath = './X_train'
Y_train_fpath = './Y_train'
X_test_fpath = './X_test'
output_fpath = './output_{}.csv'

# 把csv文件转换成numpy的数组
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)


def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column is None:
        specified_column = np.arange(X.shape[1])

    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)  # 1e-8防止除零

    return X, X_mean, X_std


# 标准化训练数据和测试数据
X_train, X_mean, X_std = _normalize(X_train, train=True)
X_test, _, _ = _normalize(X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)
# 用 _ 这个变量来存储函数返回的无用值


def _train_dev_split(X, Y, dev_ratio=0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


# 把数据分成训练集和验证集
dev_ratio = 0.2
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)


def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return X[randomize], Y[randomize]


def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)


def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)


def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc


def _cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy


def _gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad


def initialize_adam(w, b):
    """
    初始化v和s，它们都是字典类型的变量，都包含了以下字段：
        - keys: "dW", "db"
        - values：与对应的梯度/参数相同维度的值为零的numpy矩阵

    参数：
        w - 参数 w
        b - 参数 b
    返回：
        v - 包含梯度的指数加权平均值，字段如下：
            v["dW"] = ...
            v["db"] = ...
        s - 包含平方梯度的指数加权平均值，字段如下：
            s["dW"] = ...
            s["db"] = ...

    """

    v = {}
    s = {}

    v["dW"] = np.zeros_like(w)
    v["db"] = np.zeros_like(b)

    s["dW"] = np.zeros_like(w)
    s["db"] = np.zeros_like(b)

    return v, s


def update_parameters_with_adam(w, b, w_grad, b_grad, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    使用Adam更新参数

    参数：
        w - 参数 w
        b - 参数 b
        w_grad - 参数w的梯度
        b_grad - 参数b的梯度
        v - Adam的变量，第一个梯度的移动平均值，是一个字典类型的变量
        s - Adam的变量，平方梯度的移动平均值，是一个字典类型的变量
        t - 当前迭代的次数
        learning_rate - 学习率
        beta1 - 动量，超参数,用于第一阶段，使得曲线的Y值不从0开始（参见天气数据的那个图）
        beta2 - RMSprop的一个参数，超参数
        epsilon - 防止除零操作（分母为0）

    返回：
        w, b
    """

    # 梯度的移动平均值,输入："v , grads , beta1",输出：" v "
    v_corrected = {}  # 偏差修正后的值
    s_corrected = {}  # 偏差修正后的值

    # 梯度的移动平均值,输入："v , grads , beta1",输出：" v "
    v["dW"] = beta1 * v["dW"] + (1 - beta1) * w_grad
    v["db"] = beta1 * v["db"] + (1 - beta1) * b_grad

    # 计算第一阶段的偏差修正后的估计值，输入"v , beta1 , t" , 输出："v_corrected"
    v_corrected["dW"] = v["dW"] / (1 - np.power(beta1, t))
    v_corrected["db"] = v["db"] / (1 - np.power(beta1, t))

    # 计算平方梯度的移动平均值，输入："s, grads , beta2"，输出："s"
    s["dW"] = beta2 * s["dW"] + (1 - beta2) * np.square(w_grad)
    s["db"] = beta2 * s["db"] + (1 - beta2) * np.square(b_grad)

    # 计算第二阶段的偏差修正后的估计值，输入："s , beta2 , t"，输出："s_corrected"
    s_corrected["dW"] = s["dW"] / (1 - np.power(beta2, t))
    s_corrected["db"] = s["db"] / (1 - np.power(beta2, t))

    # 更新参数，输入: "parameters, learning_rate, v_corrected, s_corrected, epsilon". 输出: "parameters".
    w = w - learning_rate * (v_corrected["dW"] / np.sqrt(s_corrected["dW"] + epsilon))
    b = b - learning_rate * (v_corrected["db"] / np.sqrt(s_corrected["db"] + epsilon))

    return w, b, v, s


train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
# 初始化权重w和b，令它们都为0
w = np.zeros((data_dim,))  # [0,0,0,...,0]
b = np.zeros((1,))  # [0]

# 训练时的超参数
max_iter = 50
batch_size = 32
learning_rate = 0.001

# 保存每个iteration的loss和accuracy，以便后续画图
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

# adagrad所需的累加和
adagrad_w = 0
adagrad_b = 0
# 防止adagrad除零
epsilon = 1e-8

# 累计参数更新的次数
step = 1
v, s = initialize_adam(w, b)
beta1 = 0.9
beta2 = 0.999

# 迭代训练
for epoch in range(max_iter):
    # 在每个epoch开始时，随机打散训练数据
    X_train, Y_train = _shuffle(X_train, Y_train)

    # Mini-batch训练
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx * batch_size:(idx + 1) * batch_size]
        Y = Y_train[idx * batch_size:(idx + 1) * batch_size]

        # 计算梯度
        w_grad, b_grad = _gradient(X, Y, w, b)

        # 梯度下降法更新
        # 学习率随时间衰减
        # adagrad_w += w_grad ** 2
        # adagrad_b += b_grad ** 2

        # 梯度下降法adagrad更新w和b
        # w = w - learning_rate / (np.sqrt(adagrad_w + epsilon)) * w_grad
        # b = b - learning_rate / (np.sqrt(adagrad_b + epsilon)) * b_grad
        # w = w - learning_rate / np.sqrt(step) * w_grad
        # b = b - learning_rate / np.sqrt(step) * b_grad
        w, b, v, s = update_parameters_with_adam(w, b, w_grad, b_grad, v, s, step, learning_rate, beta1, beta2, epsilon)
        step = step + 1

    # 计算训练集和验证集的loss和accuracy
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)

print('Training loss: {}'.format(train_loss[-1]))
print('Development loss: {}'.format(dev_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))

# Loss曲线
plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.show()

# Accuracy曲线
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.show()

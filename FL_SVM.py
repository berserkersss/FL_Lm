#!/usr/bin/env python
# coding: utf-8

# # Fed-Learning in Wireless Environment
# 此函数用户学长w的要求生成，w初始化为0,使用学姐生成标签的方式, 10个类，使用CL和FedAvg
# ## Import Libraries


# In[1]:

import pandas as pd
import numpy as np
from sklearn import svm, metrics
import matplotlib.pyplot as plt
from models.Compute import compute_Lm


#Functions
# In[55]:
# lam is penalty
def get_loss_local(W, X, y, lam):
    losses_class = []  ### list containing loss per classifier
    N_X = X.shape[0]
    N_class = 10  # flatten 平铺为一维数据

    ### calculate loss per classifier
    ## 损失函数的选取可以根据自己需要改变，这里是按照类别进行loss function的计算
    for label in range(N_class):
        ### binarize y
        y_binary = np.copy(y)  # initialize
        y_binary[y_binary == label] = 1
        y_binary[y_binary != label] = -1

        w = W[label, :]

        loss = 0
        first_term = (lam / 2) * np.linalg.norm(w, 2)  # regularization
        for i in range(N_X):
            second_term = (1 / 2) * max(0, 1 - y_binary[i] * np.dot(w, X[i, :].T))
            loss += second_term

        loss += first_term
        loss /= N_X
        losses_class.append(loss)

    total_loss = sum(losses_class) / N_class

    return total_loss


# In[19]:

# 全局的 loss 就是本地 loss 根据数据集大小的一个加权和
# losses_local is list containing loss per device
def get_loss_global(losses_local, Xs_train):
    loss = 0
    N_data = 0
    K = len(Xs_train)  # number of devices
    for k in range(K):
        loss += Xs_train[k].shape[0] * losses_local[k]
        N_data += Xs_train[k].shape[0]

    loss /= N_data

    return loss


# In[21]:

### 根据loss function 计算梯度下降的表达式，将
### get sum of gradient per class
### w is 1-D
def get_sum_gradient(w, X, y, label, lam):
    ### binarize y
    y_binary = np.copy(y)  # initialize
    y_binary[y == label] = 1
    y_binary[y != label] = -1

    N_X = X.shape[0]
    sum_gradient = 0

    first_term = lam * w
    for i in range(N_X):
        indicator = 1 - y_binary[i] * np.dot(w, X[i, :].T)
        if (indicator <= 0):
            second_term = 0
        else:
            second_term = (-y_binary[i] * X[i, :])  # w is 1*n_features vector
        sum_gradient += first_term + second_term

    sum_gradient /= N_X

    return sum_gradient


# In[22]:

# 根据计算的梯度更新 w
# each row of W means all-vs-one classifier
def update_weight(W, X, y, lam, eta):
    W_updated = np.copy(W)  # initialize
    N_class = 10
    grad_class = []
    for label in range(N_class):
        w = W[label, :]
        grad = get_sum_gradient(w, X, y, label, lam)
        W_updated[label, :] = w - eta * grad
        grad_class.append(grad)

    return W_updated


# In[93]:

# 按照每个设备运算的数据量大小对 w_local 进行加权求和
# Xs_train is list containing dataset per device, Ws_local is list containing the coordinates of the center per device
# this function is for unstable connection although global updates are distributed to all devices because server has an enough power to do that
def limited_global(Xs_train, Ws_local):
    K = len(Ws_local)  # number of devices
    K_limited = len(Xs_train)  # number of devices to get update
    N_data = 0  # initialize number of the whole data
    W_global = np.zeros(Ws_local[0].shape)  # initialize

    for k in range(K_limited):
        N_X_train = Xs_train[k].shape[0]
        W_global += N_X_train * Ws_local[k]
        N_data += N_X_train
    W_global /= N_data
    Ws_local_updated = [W_global for k in
                        range(K)]  # update local weights based on global weight are distributed to all the devices
    return W_global, Ws_local_updated


def limited_global_optimize(Ld, Ws_local):
    K = len(Ws_local)  # number of devices
    W_global = np.zeros(Ws_local[0].shape)  # initialize

    for k in range(len(Ld)):
        W_global += Ld[k] * Ws_local[k]
    Ws_local_updated = [W_global for k in
                        range(K)]  # update local weights based on global weight are distributed to all the devices

    return W_global, Ws_local_updated


def predict(W, X):
    Y = np.dot(W, X.T)  # Y : number of class * number of X
    y_predict = np.argmax(Y, axis=0)  # find the max value in each row(axis=0)

    return y_predict



# 用户数据集大小和标签种类
D_dis = [0.9341333333333343, 0.8510333333333331, 0.7378333333333333, 0.7176999999999998, 0.45223333333333315]

num_img = [1200, 600, 600, 600, 400]
num_label = [2, 1, 3, 2, 8]
#Ld = [0.0612, 0.0335, 0.2023, 0.0566, 0.6486]

T = 10
tau = 5
print("you entered T=%r, tau=%r" % (T, tau))
eta = 0.01  # step size of gradient descent
lam = 0.01  # 代表了正则项的影响因子，取值越大，模型复杂度对最终代价函数的惩罚力度就越大
num_device = 5  # total number of devices to be selected
num_Lm = 6

# ## Hyperparameters
csv_path_train_data = 'csv/training_image.csv'
csv_path_train_label = 'csv/training_label.csv'

X_all_train = pd.read_csv(csv_path_train_data, header=None)
y_all_train = pd.read_csv(csv_path_train_label, header=None)
X_all_train = X_all_train.values
y_all_train = y_all_train.values

# ## Hyperparameters
csv_path_test_data = 'csv/test_image.csv'
csv_path_test_label = 'csv/test_label.csv'

X_all_test = pd.read_csv(csv_path_test_data, header=None)
y_all_test = pd.read_csv(csv_path_test_label, header=None)

# ## Scaling Data
# Scale down the magnitude for better traning experience
# 源图片每个像素有255个灰度级，对图片数据进行放缩处理有助于提高运算速度
X_all_test /= 255

X_all_test = X_all_test.values
y_all_test = y_all_test.values
# In[145]:
n_test = 1000
indexes_test_batch = np.random.choice(range(X_all_test.shape[0]), size=n_test, replace=False)

X_test = X_all_test[indexes_test_batch, :]
y_test = y_all_test[indexes_test_batch]

max_accuracy_fl, max_accuracy_cl = [], []

filename = 'result/SVM/' + "Accuracy_FedAvg_MAX_SVM" + ".csv"
with open(filename, "w") as myfile:
    myfile.write(str(0) + ',')
filename = 'result/SVM/' + "Accuracy_FedAvg_Optimize_MAX_SVM" + " .csv"
with open(filename, "w") as myfile:
    myfile.write(str(0) + ',')
for num_itr in range(num_Lm):
    X_train_user = []
    dict_users, dict_users_balance = {}, {}
    for k in range(len(num_img)):
        #  导入unbalance数据集
        csv_path_train_data = 'user_csv/index/' + 'user' + str(k) + 'train_index_' + '.csv'
        train_index = pd.read_csv(csv_path_train_data, header=None)

        # 修剪数据集使得只有图片和标签,把序号剔除
        train_index = train_index.values.astype(int)
        dict_users[k] = (train_index[num_itr, :])
        X_train_user.append(X_all_train[train_index[num_itr, :], :])
    Ld = compute_Lm(num_img, X_train_user)

    X_all_train /= 255
    X_train, y_train = [], []
    for k in range(num_device):
        X_train.append(X_all_train[dict_users[k], :])
        y_train.append(y_all_train[dict_users[k]])


    # 有多少个y的不同的值就有多少个 w 分类器,分类器的长度等于图像的大小28*28
    # initialize weight(global and local), Ws_local is list containing weight per device
    # each row of W means all-vs-one classifier
    # The set function eleminates the repeated elements, only the tags are remained
    N_class = len(set(y_all_test.flatten()))
    N_features = X_all_test.shape[1]

    # For each class there is a specific feature, in this example, it is 8*784
    Ws_global_Fvg = np.random.rand(N_class, N_features) * 0.1
    Ws_local_Fvg = [Ws_global_Fvg for k in range(num_device)]

    Ws_global_Fvg_Optimize = Ws_global_Fvg
    Ws_local_Fvg_Optimize = [Ws_global_Fvg for k in range(num_device)]

    filename = 'model/' + "model.csv"
    np.savetxt(filename, Ws_global_Fvg)

    # ## Simulations

    # In[150]:
    # 新建存放数据的文件, 并存放初始数据
    filename = 'result/SVM/' + "Accuracy_FedAvg_SVM" + str(num_itr) + ".csv"
    with open(filename, "w") as myfile:
        myfile.write(str(0) + ',')
    filename = 'result/SVM/' + "Accuracy_FedAvg_Optimize_SVM" + str(num_itr) + " .csv"
    with open(filename, "w") as myfile:
        myfile.write(str(0) + ',')

    acc_train_cl_his, acc_train_fl_his, acc_train_cl_his_iid = [], [], []
    acc_train_cl_his2, acc_train_fl_his2 = [], []
    for t in range(T):
        if T > 0:
            for k in range(num_device):
                # 更新FL的w
                Ws_local_Fvg[k]= update_weight(Ws_local_Fvg[k], X_train[k], y_train[k], lam,
                                                         eta)
                Ws_local_Fvg_Optimize[k] = update_weight(Ws_local_Fvg_Optimize[k], X_train[k], y_train[k], lam,
                                                  eta)

            # Upon each update slot, select the devices and udpate the weight
            if t % tau == 0:
                print('It is the %r-th step out of %r' % (t / tau + 1, t / tau))

                y_predict = predict(Ws_global_Fvg, X_test)
                accuracy = metrics.accuracy_score(y_test, y_predict)
                print("FL: %f" % accuracy)
                acc_train_fl_his.append(accuracy)
                filename = 'result/SVM/' + "Accuracy_FedAvg_SVM" + str(num_itr) + ".csv"
                with open(filename, "a") as myfile:
                    myfile.write(str(accuracy) + ',')

                y_predict = predict(Ws_global_Fvg_Optimize, X_test)
                accuracy = metrics.accuracy_score(y_test, y_predict)
                print("FL_Optimize: %f" % accuracy)
                acc_train_cl_his.append(accuracy)
                filename = 'result/SVM/' + "Accuracy_FedAvg_Optimize_SVM" + str(num_itr) + " .csv"
                with open(filename, "a") as myfile:
                    myfile.write(str(accuracy) + ',')

                # Fvg算法
                Ws_global_Fvg, Ws_local_Fvg = limited_global(X_train, Ws_local_Fvg)
                Ws_global_Fvg_Optimize, Ws_local_Fvg_Optimize = limited_global_optimize(Ld, Ws_local_Fvg_Optimize)

    colors = ["navy", "red"]
    labels = ["FedAvg_unbalance", "FedAvg_Optimize_unbalance"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(acc_train_fl_his, c=colors[0], label=labels[0])
    ax.plot(acc_train_cl_his, c=colors[1], label=labels[1])
    ax.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.savefig('Figures/Accuracy_SVM'+str(num_itr)+'.png')
    plt.savefig('Figures/Accuracy_SVM'+str(num_itr)+'.eps')

    max_accuracy_cl.append(max(acc_train_cl_his))
    max_accuracy_fl.append(max(acc_train_fl_his))

    filename = 'result/SVM/' + "Accuracy_FedAvg_MAX_SVM" + ".csv"
    with open(filename, "a") as myfile:
        myfile.write(str(max(acc_train_fl_his)) + ',')
    filename = 'result/SVM/' + "Accuracy_FedAvg_Optimize_MAX_SVM" + " .csv"
    with open(filename, "a") as myfile:
        myfile.write(str(max(acc_train_cl_his)) + ',')


colors = ["navy", "red"]
labels = ["FedAvg_unbalance", "FedAvg_Optimize_unbalance"]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(max_accuracy_fl, c=colors[0], label=labels[0])
ax.plot(max_accuracy_cl, c=colors[1], label=labels[1])
ax.legend()
plt.xlabel('Lm')
plt.ylabel('Accuracy')
plt.savefig('Figures/Accuracy_MAX.png')
plt.savefig('Figures/Accuracy_MAX.eps')



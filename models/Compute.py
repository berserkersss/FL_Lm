#!/usr/bin/env python
# coding: utf-8

# # Fed-Learning in Wireless Environment
# 此函数用于构造所需数据集，并计算P(x)
# ## Import Libraries

# In[1]:

import pandas as pd
import numpy as np


def compute_px(X_all_train, X_train_user):
    Px_all = np.zeros((1, X_all_train.shape[1]))
    Px_user = np.zeros((5, X_all_train.shape[1]))

    Px_all_gray = np.arange(784)
    Px_user_gray = {i: np.arange(784) for i in range(len(X_train_user))}
    for gray in range(256):
        for i in range(X_all_train.shape[1]):
            Px_all[0, i] = np.sum(X_all_train[:, i] == gray)
            Px_all[0, i] = Px_all[0, i] / 60000.0
            for j in range(len(X_train_user)):
                temp = X_train_user[j]
                Px_user[j, i] = np.sum(temp[:, i] == gray)
                Px_user[j, i] = 1.0 * Px_user[j, i] / len(temp[:, 0])
        for j in range(len(X_train_user)):
            Px_user_gray[j] = np.vstack((Px_user_gray[j], Px_user[j, :]))
        Px_all_gray = np.vstack((Px_all_gray, Px_all))

    Px_all_gray = Px_all_gray[1:, :]
    D_distribution = []
    D = []
    for idx in range(len(X_train_user)):
        Px = Px_user_gray[idx][1:, :]
        diff = Px_all_gray - Px
        D_distribution.append(np.linalg.norm(diff, ord=1))
        temp = np.linalg.norm(diff, ord=2, axis=0)
        D.append(np.linalg.norm(temp, ord=1))

    # print(D_distribution)
    print(D)


def compute_Lm(num_img, X_train_user):
    #compute P(x)
    csv_path_train_data = 'csv/training_image.csv'
    X_all_train = pd.read_csv(csv_path_train_data, header=None)
    X_all_train = X_all_train.values

    Px_all = np.zeros((1, X_all_train.shape[1]))
    Px_user = np.zeros((5, X_all_train.shape[1]))

    Px_all_gray = np.arange(784)
    Px_user_gray = {i: np.arange(784) for i in range(len(X_train_user))}
    for gray in range(256):
        for i in range(X_all_train.shape[1]):
            Px_all[0, i] = np.sum(X_all_train[:, i] == gray)
            Px_all[0, i] = Px_all[0, i] / 60000.0
            for j in range(len(X_train_user)):
                temp = X_train_user[j]
                Px_user[j, i] = np.sum(temp[:, i] == gray)
                Px_user[j, i] = 1.0 * Px_user[j, i] / len(temp[:, 0])
        for j in range(len(X_train_user)):
            Px_user_gray[j] = np.vstack((Px_user_gray[j], Px_user[j, :]))
        Px_all_gray = np.vstack((Px_all_gray, Px_all))

    Px_all_gray = Px_all_gray[1:, :]
    D_distribution = []
    D = []
    for idx in range(len(X_train_user)):
        Px = Px_user_gray[idx][1:, :]
        diff = Px_all_gray - Px
        D_distribution.append(np.linalg.norm(diff, ord=1))
        temp = np.linalg.norm(diff, ord=2, axis=0)
        D.append(np.linalg.norm(temp, ord=1))

    # print(D_distribution)
    print(D)

    # compute Ld
    mu = 3.5169
    sigma = 0.2405
    Lm = []
    Ld = []
    for i in range(len(num_img)):
        Lm.append(1/(4*sigma**2/num_img[i] + 2*(mu*D[i])**2))
    for i in range(len(num_img)):
        Ld.append(Lm[i]/(sum(Lm)))
    return Ld


if __name__ == '__main__':
    # 用户数据集大小和标签种类
    num_img = [600, 600, 600, 600, 600]
    num_label = [2, 2, 2, 4, 8]
    num_Sim = 6

    csv_path_train_data = 'csv/training_image.csv'
    X_all_train = pd.read_csv(csv_path_train_data, header=None)
    X_all_train = X_all_train.values

    csv_path_train_data = 'csv/training_label.csv'
    y_all_train = pd.read_csv(csv_path_train_data, header=None)
    y_all_train = y_all_train.values

    for i in range(num_Sim):
        X_train_user = []
        y_train_user = []
        for k in range(len(num_img)):
            csv_path_train_data = 'user_csv/index/' + 'user' + str(k) + 'train_index_' + '.csv'
            train_index = pd.read_csv(csv_path_train_data, header=None)

            # 修剪数据集使得只有图片和标签,把序号剔除
            train_index = train_index.values
            train_index = train_index.astype(int)
            X_train_user.append(X_all_train[train_index[i, :], :])
            y_train_user.append(X_all_train[train_index[i, :], :])
        compute_px(X_all_train, X_train_user)

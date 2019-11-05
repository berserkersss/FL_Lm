#!/usr/bin/env python
# coding: utf-8

# # Fed-Learning in Wireless Environment
# 此函数用于构造所需数据集，并计算P(x)
# ## Import Libraries

# In[1]:

import pandas as pd
import numpy as np
from sklearn import svm, metrics
import math
import matplotlib.pyplot as plt


def mnist_noniid(labels, num_label, num_img):
    """
    Sample non-I.I.D client data from MNIST dataset
    """
    total = 60000
    num_shards, num_imgs = int(total * num_label / num_img), int(num_img / num_label)
    idx_shard = [i for i in range(num_shards)]
    dict_users = np.array([], dtype='int64')
    idxs = np.arange(num_shards * num_imgs)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    rand_set = set(np.random.choice(idx_shard, num_label, replace=False))  # 50
    for rand in rand_set:
        dict_users = np.concatenate((dict_users, idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    return dict_users


if __name__ == '__main__':
    csv_path_train_data = 'csv/training_image.csv'
    csv_path_train_label = 'csv/training_label.csv'

    # 设置用户数据集大小和标签种类
    num_img = [1200, 600, 600, 600, 400]
    num_labels = []
    for i in range(6):
        num_label = np.array([1, 1, 1, 1]) * (i+1)
        num_label = num_label.tolist()
        num_label.append(8)
        num_labels.append(num_label)
    # 导入MNIST数据集，数据集由 60,000 个训练样本和 10,000 个测试样本组成，每个样本
    # 为一个28*28的图片，读入时我们将这个图片转换为1*784的向量
    # header=None 表示文件一开始就是数据
    X_all_train = pd.read_csv(csv_path_train_data, header=None)
    y_all_train = pd.read_csv(csv_path_train_label, header=None)

    # In[3]:
    X_all_train = X_all_train.values
    y_all_train = y_all_train.values

    for k in range(len(num_img)):
        filename_idx = 'user_csv/index/' + 'user' + str(k) + 'train_index_' + '.csv'
        for i in range(len(num_labels)):
            while True:
                index_set = mnist_noniid(y_all_train.T, num_labels[i][k], num_img[k])

                X_train = X_all_train[index_set, :]
                y_train = y_all_train[index_set, :]
                if len(set(y_train.flatten())) == num_labels[i][k]:  # 保证标签的个数，否则重新抽取
                    if i == 0:
                        index_sets = index_set
                    else:
                        index_sets = np.row_stack((index_sets, index_set))
                    break
        np.savetxt(filename_idx, index_sets, delimiter=',')

        # filename = 'user_csv/' + 'user' + str(k) + 'train_label' + label + '.csv'
        # filename1 ='user_csv/' + 'user' + str(k) + 'train_img' + label + '.csv'
        # np.savetxt(filename1, X_train, delimiter=',')
        # np.savetxt(filename, y_train, delimiter=',')

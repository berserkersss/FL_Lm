#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# 这个用于CNN的不同分布差异数据仿真, 手写体只需要一个通道
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import math

from models.Compute import compute_Lm
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Update import CLUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.Fed import FedAvg_Optimize
from models.test import test_img

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=False, transform=trans_mnist)
    dataset_test = datasets.MNIST('../data/mnist/', train=False, download=False, transform=trans_mnist)

    # sample users

    num_img = [1200, 600, 600, 600, 400]
    num_label = [2, 1, 3, 2, 8]
    num_Lm = 6

    filename = 'result/CNN_D/' + "Accuracy_FedAvg_MAX_CNN" + ".csv"
    with open(filename, "w") as myfile:
        myfile.write(str(0) + ',')
    filename = 'result/CNN_D/' + "Accuracy_FedAvg_Optimize_MAX_CNN" + " .csv"
    with open(filename, "w") as myfile:
        myfile.write(str(0) + ',')
    net_glob = CNNMnist(args=args).to(args.device)

    csv_path_train_data = 'csv/training_image.csv'
    X_all_train = pd.read_csv(csv_path_train_data, header=None)
    X_all_train = X_all_train.values

    max_accuracy_cl, max_accuracy_fl = [], []
    for num_itr in range(num_Lm):
        dict_users = {}
        X_train_user = []
        for k in range(len(num_img)):
            #  导入unbalance数据集
            csv_path_train_data = 'user_csv/index/' + 'user' + str(k) + 'train_index_' + '.csv'
            train_index = pd.read_csv(csv_path_train_data, header=None)

            # 修剪数据集使得只有图片和标签,把序号剔除
            train_index = train_index.values
            train_index = train_index.astype(int)
            dict_users[k] = np.array(train_index[num_itr, :])
            X_train_user.append(X_all_train[train_index[num_itr, :], :])
        Ld = compute_Lm(num_img, X_train_user)

        net_glob_fl = copy.deepcopy(net_glob)
        net_glob_cl = copy.deepcopy(net_glob)

        net_glob_fl.train()
        net_glob_cl.train()

        # copy weights
        w_glob_fl = net_glob_fl.state_dict()
        w_glob_cl = net_glob_cl.state_dict()

        acc_train_cl_his, acc_train_fl_his, acc_train_cl_his_iid = [], [], []
        acc_train_cl_his2, acc_train_fl_his2 = [], []

        # 新建存放数据的文件
        filename = 'result/CNN_D/' + "Accuracy_FedAvg_CNN" + str(num_itr) + ".csv"
        np.savetxt(filename, [])
        filename = 'result/CNN_D/' + "Accuracy_FedAvg_Optimize_CNN" + str(num_itr) + ".csv"
        np.savetxt(filename, [])

        for iter in range(args.epochs):  # num of iterations

            # FL setting
            # testing
            net_glob_fl.eval()
            acc_test_fl, loss_test_flxx = test_img(net_glob_fl, dataset_test, args)
            print("Testing accuracy FL: {:.2f}".format(acc_test_fl))
            acc_train_fl_his.append(acc_test_fl)

            filename = 'result/CNN_D/' + "Accuracy_FedAvg_CNN" + str(num_itr) + ".csv"
            with open(filename, "a") as myfile:
                myfile.write(str(acc_test_fl) + ',')

            w_locals, loss_locals = [], []
            # M clients local update
            m = max(int(args.frac * args.num_users), 1)  # num of selected users
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # select randomly m clients
            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])  # data select
                w, loss = local.train(net=copy.deepcopy(net_glob_fl).to(args.device))
                w_locals.append(copy.deepcopy(w))  # collect local model
                loss_locals.append(copy.deepcopy(loss))  # collect local loss fucntion

            w_glob_fl = FedAvg(w_locals)  # update the global model
            net_glob_fl.load_state_dict(w_glob_fl)  # copy weight to net_glob


            # FL_Optimize setting
            # testing
            net_glob_cl.eval()
            acc_test_cl, loss_test_clxx = test_img(net_glob_cl, dataset_test, args)
            print("Testing accuracy FL_optimize: {:.2f}".format(acc_test_cl))
            acc_train_cl_his.append(acc_test_cl)

            filename = 'result/CNN_D/' + "Accuracy_FedAvg_Optimize_CNN" + str(num_itr) + ".csv"
            with open(filename, "a") as myfile:
                myfile.write(str(acc_test_cl) + ',')

            w_locals, loss_locals = [], []
            # M clients local update
            for idx in range(args.num_users):
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])  # data select
                w, loss = local.train(net=copy.deepcopy(net_glob_cl).to(args.device))
                w_locals.append(copy.deepcopy(w))  # collect local model
                loss_locals.append(copy.deepcopy(loss))  # collect local loss fucntion

            w_glob_cl = FedAvg_Optimize(w_locals, Ld)  # update the global model
            net_glob_cl.load_state_dict(w_glob_cl)  # copy weight to net_glob

        colors = ["navy", "red"]
        labels = ["FedAvg_S", "FedAvg_Optimize_S"]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(acc_train_fl_his, c=colors[0], label=labels[0])
        ax.plot(acc_train_cl_his, c=colors[1], label=labels[1])
        ax.legend()
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.savefig('Figures/Accuracy_CNN' + str(num_itr) + '.png')

        filename = 'result/CNN_D/' + "Accuracy_FedAvg_MAX_CNN" + ".csv"
        with open(filename, "a") as myfile:
            myfile.write(str(max(acc_train_fl_his)) + ',')
        filename = 'result/CNN_D/' + "Accuracy_FedAvg_Optimize_MAX_CNN" + " .csv"
        with open(filename, "a") as myfile:
            myfile.write(str(max(acc_train_cl_his)) + ',')

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

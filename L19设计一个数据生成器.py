# -*- coding: UTF-8-*-
"""
@Project: LearnTorch
@Author: Oops404
@Email: cheneyjin@outlook.com
@Time: 2022/1/24 14:18
"""
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader

"""
炼丹师(ง •_•)ง的成长之路，生成器的目的是为了控制变量。
"""


# # 输入为2个特征，1000条样本
# num_inputs = 2
# num_examples = 1000
#
# torch.manual_seed(996)
#
# """
# 构建方程 y=2x_1 - x_2 + 1
# """
# # w系数
# coef_w = torch.tensor([2., -1]).reshape(2, 1)
# # b截距
# coef_b = torch.tensor(1.)
#
# features = torch.randn(num_examples, num_inputs)
# # 构建完成
# labels_true = torch.mm(features, coef_w) + coef_b
# # 撒上噪声,randn生成的随机数本身满足正态分布
# labels = labels_true + torch.randn(size=labels_true.shape) * 0.01
#
# # # 绘制一下，121：12表示一行两列，1代表绘制第1个
# # plt.subplot(121)
# # plt.scatter(features[:, 0], labels)
# # # 2代表绘制第2个
# # plt.subplot(122)
# # plt.scatter(features[:, 1], labels)
# # plt.show()
#
# labels1 = labels_true + torch.randn(size=labels_true.shape) * 2
#
# plt.subplot(221)
# plt.scatter(features[:, 0], labels)
# # 2代表绘制第2个
# plt.subplot(222)
# plt.plot(features[:, 1], labels, 'ro')
#
# plt.subplot(223)
# plt.scatter(features[:, 0], labels1)
# # 2代表绘制第2个
# plt.subplot(224)
# plt.plot(features[:, 1], labels1, 'yo')
#
# plt.show()


def data_reg_generator(num_examples=1000, w=None, bias=True, delta=0.01, deg=1):
    """
    综上思路，设计回归类数据生成器:
    :param num_examples: 数据量
    :param w: 自变量系数关系
    :param bias: 截距
    :param delta: 扰动项参数
    :param deg: 方程次数
    :return: 生成特征张量和标签张量
    """
    if w is None:
        w = [2, -1, 1]
    if bias:
        num_inputs = len(w) - 1
        features_true = torch.randn(num_examples, num_inputs)
        w_true = torch.tensor(w[:-1]).reshape(-1, 1).float()
        b_true = torch.tensor(w[-1]).float()
        if num_inputs == 1:
            labels_true = torch.pow(features_true, deg) * w_true + b_true
        else:
            # 避免非常复杂，对于交叉项有不足，if deg!=1
            labels_true = torch.mm(torch.pow(features_true, deg), w_true) + b_true
        features = torch.cat((features_true, torch.ones(len(features_true), 1)), 1)
        labels = labels_true + torch.randn(size=labels_true.shape) * delta
    else:
        num_inputs = len(w)
        features = torch.randn(num_examples, num_inputs)
        w_true = torch.tensor(w).reshape(-1, 1).float()
        if num_inputs == 1:
            labels_true = torch.pow(features, deg) * w_true
        else:
            labels_true = torch.mm(torch.pow(features, deg), w_true)
        labels = labels_true * torch.randn(size=labels_true.shape) * delta
    return features, labels


torch.manual_seed(996)
_features, _labels = data_reg_generator(deg=2, delta=0.15)
plt.subplot(121)
plt.scatter(_features[:, 0], _labels)
plt.subplot(122)
plt.scatter(_features[:, 1], _labels)
plt.show()

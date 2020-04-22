# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import misc


def tpr95(name):
    # calculate the falsepositive error when tpr is 95%
    # calculate baseline
    T = 1
    cifar = np.loadtxt('./softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Base_Out.txt', delimiter=',')

    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    start = min(np.min(X1), np.min(Y1))
    end = max(np.max(X1), np.max(Y1))
    gap = (end - start) / 100000
    total = 0.0
    fpr = 0.0
    max_tpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr > max_tpr: max_tpr = tpr
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1

    fprBase = -1.0
    if total > 0:
        fprBase = fpr / total
    else:
        print('max tpr ', max_tpr, ' !')
        return fprBase, -1.0

    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('./softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Our_Out.txt', delimiter=',')

    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    start = min(np.min(X1), np.min(Y1))
    end = max(np.max(X1), np.max(Y1))
    gap = (end - start) / 100000
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1

    fprNew = -1.0
    if total > 0:
        fprNew = fpr / total
    else:
        print('max tpr ', max_tpr, ' !')
        return fprBase, fprNew

    return fprBase, fprNew


def cal_auroc(in_file_path, out_file_path):
    in_data = np.loadtxt(in_file_path, delimiter=',')
    out_data = np.loadtxt(out_file_path, delimiter=',')
    in_softmax_scores = in_data[:, 2]
    out_softmax_scores = out_data[:, 2]
    start = min(np.min(in_softmax_scores), np.min(out_softmax_scores))
    end = max(np.max(in_softmax_scores), np.max(out_softmax_scores))
    # print('auroc in', start, ',', end)
    gap = (end - start) / 100000
    aurocValue = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(in_softmax_scores >= delta)) / np.float(len(in_softmax_scores))
        fpr = np.sum(np.sum(out_softmax_scores > delta)) / np.float(len(out_softmax_scores))
        aurocValue += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocValue += fpr * tpr
    return aurocValue


def auroc(name):
    # calculate the AUROC
    # AUROC:                       95.4%              98.4%
    return cal_auroc('./softmax_scores/confidence_Base_In.txt', './softmax_scores/confidence_Base_Out.txt'), \
           cal_auroc('./softmax_scores/confidence_Our_In.txt', './softmax_scores/confidence_Our_Out.txt')


def auprIn(name):
    # calculate the AUPR
    # calculate baseline
    T = 1
    cifar = np.loadtxt('./softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Base_Out.txt', delimiter=',')

    precisionVec = []
    recallVec = []
    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    start = min(np.min(X1), np.min(Y1))
    end = max(np.max(X1), np.max(Y1))
    gap = (end - start) / 100000
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision
    # print(recall, precision)

    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('./softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Our_Out.txt', delimiter=',')

    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    start = min(np.min(X1), np.min(Y1))
    end = max(np.max(X1), np.max(Y1))
    gap = (end - start) / 100000
    auprNew = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        # precisionVec.append(precision)
        # recallVec.append(recall)
        auprNew += (recallTemp - recall) * precision
        recallTemp = recall
    auprNew += recall * precision
    return auprBase, auprNew


def auprOut(name):
    # calculate the AUPR
    # calculate baseline
    T = 1
    cifar = np.loadtxt('./softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Base_Out.txt', delimiter=',')

    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    start = min(np.min(X1), np.min(Y1))
    end = max(np.max(X1), np.max(Y1))
    gap = (end - start) / 100000
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprBase += (recallTemp - recall) * precision
        recallTemp = recall
    auprBase += recall * precision

    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('./softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Our_Out.txt', delimiter=',')

    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    start = min(np.min(X1), np.min(Y1))
    end = max(np.max(X1), np.max(Y1))
    gap = (end - start) / 100000
    auprNew = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprNew += (recallTemp - recall) * precision
        recallTemp = recall
    auprNew += recall * precision
    return auprBase, auprNew


def detection(name):
    # calculate the minimum detection error
    # calculate baseline
    T = 1
    cifar = np.loadtxt('./softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Base_Out.txt', delimiter=',')

    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    start = min(np.min(X1), np.min(Y1))
    end = max(np.max(X1), np.max(Y1))
    gap = (end - start) / 100000
    errorBase = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr + error2) / 2.0)

    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('./softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./softmax_scores/confidence_Our_Out.txt', delimiter=',')

    # f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    start = min(np.min(X1), np.min(Y1))
    end = max(np.max(X1), np.max(Y1))
    gap = (end - start) / 100000
    errorNew = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorNew = np.minimum(errorNew, (tpr + error2) / 2.0)

    return errorBase, errorNew


def cal_metric(in_file_path, out_file_path):
    in_data = np.loadtxt(in_file_path, delimiter=',')
    out_data = np.loadtxt(out_file_path, delimiter=',')
    in_softmax_scores = in_data[:, 2]
    out_softmax_scores = out_data[:, 2]
    start = min(np.min(in_softmax_scores), np.min(out_softmax_scores))
    end = max(np.max(in_softmax_scores), np.max(out_softmax_scores))
    # print('auroc in', start, ',', end)
    gap = (end - start) / 100000
    # tpr95
    # auroc
    aurocValue = 0.0
    fprTemp = 1.0
    # aupr in

    # cal(in_softmax_scores,out_softmax_scores,start, end, gap)

    for delta in np.arange(start, end, gap):
        # tpr95
        tpr = np.sum(np.sum(in_softmax_scores >= delta)) / np.float(len(in_softmax_scores))
        fpr = np.sum(np.sum(out_softmax_scores > delta)) / np.float(len(out_softmax_scores))
        aurocValue += (-fpr + fprTemp) * tpr
        fprTemp = fpr
    aurocValue += fpr * tpr
    return aurocValue


def metric(nn, data):
    if nn == "densenet10" or nn == "wideresnet10": indis = "CIFAR-10"
    if nn == "densenet100" or nn == "wideresnet100": indis = "CIFAR-100"
    if nn == "densenet10" or nn == "densenet100": nnStructure = "DenseNet-BC-100"
    if nn == "wideresnet10" or nn == "wideresnet100": nnStructure = "Wide-ResNet-28-10"

    if data == "Imagenet": dataName = "Tiny-ImageNet (crop)"
    if data == "Imagenet_resize": dataName = "Tiny-ImageNet (resize)"
    if data == "LSUN": dataName = "LSUN (crop)"
    if data == "LSUN_resize": dataName = "LSUN (resize)"
    if data == "iSUN": dataName = "iSUN"
    if data == "Gaussian": dataName = "Gaussian noise"
    if data == "Uniform": dataName = "Uniform Noise"
    fprBase, fprNew = tpr95(indis)
    errorBase, errorNew = detection(indis)
    aurocBase, aurocNew = auroc(indis)
    auprinBase, auprinNew = auprIn(indis)
    auproutBase, auproutNew = auprOut(indis)
    # fprBase, errorBase,aurocBase, auprinBase, auproutBase = cal_metric()

    print("{:31}{:>22}".format("Neural network architecture:", nnStructure))
    print("{:31}{:>22}".format("In-distribution dataset:", indis))
    print("{:31}{:>22}".format("Out-of-distribution dataset:", dataName))
    print("")
    print("{:>34}{:>19}".format("Baseline", "Our Method"))
    print("{:20}{:13.1f}%{:>18.1f}% ".format("FPR at TPR 95%:", fprBase * 100, fprNew * 100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("Detection error:", errorBase * 100, errorNew * 100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUROC:", aurocBase * 100, aurocNew * 100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUPR In:", auprinBase * 100, auprinNew * 100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUPR Out:", auproutBase * 100, auproutNew * 100))


if __name__ == '__main__':
    metric("densenet10", 'Imagenet')

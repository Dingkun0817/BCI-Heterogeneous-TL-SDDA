'''
Loss: MCC + DAN
22和 3 encoder后feature对齐
'''

import sys
# sys.path.append('E:\Pycharm_BCI\Self_Study\T-TIME')
sys.path.append('/data1/ldk/T-TIME')

import torch.nn.functional as F
import random
import csv
import time
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils.network import backbone_net
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar, read_mi_combine_tar_diff, read_mi_combine_tar_ldk_avg, read_mi_combine_tar_ldk_pearson_avg
from utils.utils_1 import lr_scheduler_full, fix_random_seed, cal_acc_comb, data_loader, cal_auc_comb
from utils.loss import ClassConfusionLoss
from utils.loss import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel
from tl.utils.utils_1 import data_alignment, data_alignment_session
from utils.dataloader import data_process
from utils.auto_correlation import operation_A
from tl.utils.alg_utils import EA

import gc
import sys
import math


class NSTLoss(nn.Module):
    """like what you like: knowledge distill via neuron selectivity transfer"""
    def __init__(self):
        super(NSTLoss, self).__init__()

    def forward(self, g_s, g_t):
        return self.nst_loss(g_s, g_t)

    def nst_loss(self, f_s, f_t):
        # No need to check for shape in this case, since the inputs are already 2D
        f_s = F.normalize(f_s, dim=1)
        f_t = F.normalize(f_t, dim=1)
        return torch.mean((f_s - f_t) ** 2)

# class NSTLoss(nn.Module):
#     """like what you like: knowledge distill via neuron selectivity transfer"""
#     def __init__(self):
#         super(NSTLoss, self).__init__()
#
#     def forward(self, g_s, g_t):
#         return self.nst_loss(g_s, g_t)
#
#     def nst_loss(self, f_s, f_t):
#         s_H, t_H = f_s.shape[1], f_t.shape[1]
#         if s_H > t_H:
#             f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
#         elif s_H < t_H:
#             f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
#         else:
#             pass
#         f_s = F.normalize(f_s, dim=1)
#         f_t = F.normalize(f_t, dim=1)
#         # Compute the polynomial kernel
#         kernel_loss = self.poly_kernel(f_s, f_t)
#         return kernel_loss
#
#     def poly_kernel(self, a, b):
#         a = a.unsqueeze(1)
#         b = b.unsqueeze(2)
#         res = (a * b).sum(-1).pow(2)
#         return res


def train_target(args):

    args.max_epoch = 70
    args.lr = 0.001

    X_1, y_1, num_subjects_1, paradigm_1, sample_rate_1, ch_num_1 = data_process(args.data1)  # (1296, 22, 1001)
    X_2, y_2, num_subjects_2, paradigm_2, sample_rate_2, ch_num_2 = data_process(args.data2)  # (6520, 3, 1126)

    if args.align:
        print("-------   START EA: -------")
        X_1 = data_alignment(X_1, args.N1, args.data1)
        # X_2 = data_alignment(X_2, args.N2, args.data2)
        X_2 = data_alignment_session(X_2, args.N2, args.data2)
        print("-------   FINISH EA: -------========")

    '''
    First Stage: Use Classifier Loss Train Teacher, only with src data
    '''
    # # 创建新数组存储拼接后的数据
    # X_concatenated = np.empty((X_1.shape[0], X_1.shape[1], 1126))
    #
    # # 对每个样本的每个通道进行数据拼接
    # for i in range(X_1.shape[0]):
    #     for j in range(X_1.shape[1]):
    #         X_concatenated[i, j] = np.concatenate((X_1[i, j], X_1[i, j, :125]))
    #
    # # 输出拼接后的数据的形状
    # print("拼接后的数据形状：", X_concatenated.shape)
    # X_1 = X_concatenated


    # X_1_old = X_1
    # X_1_new = np.apply_along_axis(operation_A, axis=2, arr=X_1)
    # print("自相关后的数据形状：", X_1_new.shape)
    #
    # X_1 = np.concatenate([X_1_old, X_1_new], axis=0)
    # y_1 = np.concatenate([y_1, y_1], axis=0)
    # print("拼接后的数据形状：", X_1.shape, y_1.shape)
    #
    # '''
    # 跑的时候加对齐！！！！！ 再对齐一下
    # '''
    # if args.align:
    #     print("-------   START EA: -------")
    #     X_1 = data_alignment(X_1, args.N1, args.data1)
    #     print("-------   FINISH EA: -------========")



    # X_1 = X_1[:, :, 0:1001]
    X_2 = X_2[:, :, 0:1001]   # 放前面试试？
    dset_loaders1 = data_loader(X_1, y_1, X_2, y_2, args)

    args.feature_deep_dim = 248
    args.chn = 3
    netF, netC = backbone_net(args, return_type='001xy')
    # args.feature_deep_dim = 4960
    # netF, netC = backbone_net(args, return_type='xy')
    if args.data_env != 'local':
        netF, netC = netF.cuda(), netC.cuda()
    base_network = nn.Sequential(netF, netC)
    criterion = nn.CrossEntropyLoss()
    optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
    optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(dset_loaders1["source"])
    interval_iter = max_iter // args.max_epoch
    args.max_iter = max_iter
    iter_num = 0
    base_network.train()

    while iter_num < max_iter:
        # try:
        #     inputs_source111, labels_source111 = next(iter_source)
        # except:
        #     iter_source = iter(dset_loaders1["source"])
        #     inputs_source111, labels_source111 = next(iter_source)
        # # print(labels_source111)


        iter_source = iter(dset_loaders1["source"])
        inputs_source111, labels_source111 = next(iter_source)  # torch.Size([32, 1, 22, 1001])

        iter_num += 1

        log_str = 'Task: {}, Iter:{}/{};'. \
            format(args.task_str, int(iter_num // len(dset_loaders1["source"])),
                   int(max_iter // len(dset_loaders1["source"])))
        print(log_str)

        features_source, outputs_source = base_network(inputs_source111)

        args.non_linear = False
        args.alignment_weight = 1.0
        classifier_loss1 = criterion(outputs_source, labels_source111)

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        classifier_loss1.backward()
        optimizer_f.step()
        optimizer_c.step()


    '''
    Second Stage: 老师2学生+分类Loss
    '''
    args.max_epoch = 70
    args.lr = 0.001

    X_1 = X_1[:, [7, 9, 11], :]

    print("二阶段的数据形状：", X_1.shape)


    train_x = X_1
    train_y = y_1

    if args.idt == 0:
        test_x = X_2[0:720, :, :]
        test_y = y_2[0:720]
    if args.idt == 1:
        test_x = X_2[720:1400, :, :]
        test_y = y_2[720:1400]
    if args.idt == 2:
        test_x = X_2[1400:2120, :, :]
        test_y = y_2[1400:2120]
    if args.idt == 3:
        test_x = X_2[2120:2860, :, :]
        test_y = y_2[2120:2860]
    if args.idt == 4:
        test_x = X_2[2860:3600, :, :]
        test_y = y_2[2860:3600]
    if args.idt == 5:
        test_x = X_2[3600:4320, :, :]
        test_y = y_2[3600:4320]
    if args.idt == 6:
        test_x = X_2[4320:5040, :, :]
        test_y = y_2[4320:5040]
    if args.idt == 7:
        test_x = X_2[5040:5800, :, :]
        test_y = y_2[5040:5800]
    if args.idt == 8:
        test_x = X_2[5800:6520, :, :]
        test_y = y_2[5800:6520]

    print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    dset_loaders2 = data_loader(train_x, train_y, test_x, test_y, args)

    args.feature_deep_dim = 248
    args.chn = args.chn2
    netFF, netCC = backbone_net(args, return_type='xy')
    if args.data_env != 'local':
        netFF, netCC = netFF.cuda(), netCC.cuda()
    base_network_stu = nn.Sequential(netFF, netCC)
    criterion = nn.CrossEntropyLoss()
    optimizer_ff = optim.Adam(netFF.parameters(), lr=args.lr)
    optimizer_cc = optim.Adam(netCC.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(dset_loaders2["source"])
    interval_iter = max_iter // args.max_epoch
    args.max_iter = max_iter
    iter_num = 0
    base_network_stu.train()

    while iter_num < max_iter:
        # try:
        #     inputs_source111, labels_source111 = next(iter_source)
        #     inputs_source, labels_source = next(iter_source2)
        # except:
        #     iter_source = iter(dset_loaders1["source"])
        #     inputs_source111, labels_source111 = next(iter_source)
        #     iter_source2 = iter(dset_loaders2["source"])
        #     inputs_source, labels_source = next(iter_source2)
        # # print(labels_source)
        # # print(labels_source111)
        # try:
        #     inputs_target, _ = next(iter_target)
        # except:
        #     iter_target = iter(dset_loaders2["target"])
        #     inputs_target, _ = next(iter_target)

        iter_source = iter(dset_loaders2["source"])
        inputs_source, labels_source = next(iter_source)  # torch.Size([32, 1, 22, 1001])
        iter_target = iter(dset_loaders2["target"])
        inputs_target, _ = next(iter_target)

        iter_num += 1

        log_str = 'Task: {}, Iter:{}/{};'. \
            format(args.task_str, int(iter_num // len(dset_loaders2["source"])),
                   int(max_iter // len(dset_loaders2["source"])))
        print(log_str)

        teacher_source, outputs_source_teacher = base_network(inputs_source111)   # ++
        features_source, outputs_source = base_network_stu(inputs_source)
        features_target, outputs_target = base_network_stu(inputs_target)

        args.non_linear = False
        classifier_loss = criterion(outputs_source, labels_source)

        mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
            linear=not args.non_linear
        )
        # alignment_loss1 = mkmmd_loss(teachers_feature, features_source)
        alignment_loss2 = mkmmd_loss(features_target, features_source)

        loss_function = nn.SmoothL1Loss()  # MSE没这个好
        # 计算损失
        alignment_loss1 = loss_function(teacher_source, features_source)



        '''
        试一试蒸馏Loss
        '''
        # soft_loss = nn.KLDivLoss(reduction='batchmean')
        # ditillation_loss = soft_loss(
        #     F.softmax(outputs_source_teacher / 2, dim=1),
        #     F.softmax(outputs_source / 2, dim=1)
        # )

        # def distillation(y, labels, teacher_scores, temp, alpha):
        #     return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
        #             temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)
        def distillation(y, labels, teacher_scores, temp, alpha):
            # print(teacher_scores)
            # print(y.argmax(1))
            # print(teacher_scores.argmax(1))
            # print(y.argmax(1) == teacher_scores.argmax(1))
            return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
                    temp * temp * 2.0 * alpha)

        ditillation_loss = distillation(outputs_source, labels_source, outputs_source_teacher, temp=6, alpha=0.375)   # 5  0.7

        nst_criterion = NSTLoss()

        # 在循环中使用 NSTLoss 计算蒸馏损失
        nst_loss = nst_criterion(features_source, teacher_source)
        nst_loss = torch.mean(nst_loss)
        '''
        MCC最小类混淆Loss
        '''
        args.t_mcc = 3
        transfer_loss = ClassConfusionLoss(t=args.t_mcc)(outputs_target)

        # total_loss = classifier_loss + alignment_loss + ditillation_loss
        # total_loss = classifier_loss + ditillation_loss + alignment_loss2 + 1.7 * transfer_loss   # 1  4
        total_loss = classifier_loss + 1.3 * ditillation_loss + alignment_loss2 + 1.7 * transfer_loss   # 2
        # total_loss = classifier_loss + transfer_loss     # 3
        # total_loss = classifier_loss + 1.4 * alignment_loss1 + alignment_loss2 + ditillation_loss + 1.7 * transfer_loss
        # print(classifier_loss)
        # print(alignment_loss2)
        # print(transfer_loss)
        # print(nst_loss)


        optimizer_ff.zero_grad()
        optimizer_cc.zero_grad()
        total_loss.backward()
        optimizer_ff.step()
        optimizer_cc.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network_stu.eval()

            acc_t_te, pre_t_te, rec_t_te, f1_t_te, _ = cal_acc_comb(dset_loaders2["Target"], base_network_stu, args=args)
            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%  Pre = {:.2f}%  Rec = {:.2f}%  F1 = {:.2f}%'. \
                format(args.task_str, int(iter_num // len(dset_loaders2["source"])),
                       int(max_iter // len(dset_loaders2["source"])), acc_t_te, pre_t_te, rec_t_te, f1_t_te)
            args.log.record(log_str)
            print(log_str)
            base_network_stu.train()


    # '''
    # Transfer Stage
    # '''
    # X_src, y_src, X_tar, y_tar = read_mi_combine_tar_diff(args)
    # print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    #
    # dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)
    #
    # args.max_epoch = 5
    #
    # max_iter = args.max_epoch * len(dset_loaders["source"])
    # interval_iter = max_iter // args.max_epoch
    # args.max_iter = max_iter
    # iter_num = 0
    #
    # while iter_num < max_iter:
    #     iter_source = iter(dset_loaders["source"])
    #     inputs_source, labels_source = next(iter_source)
    #
    #     iter_target = iter(dset_loaders["target"])
    #     inputs_target, _ = next(iter_target)
    #
    #     iter_num += 1
    #
    #     features_source, outputs_source = base_network_stu(inputs_source)
    #     features_target, outputs_target = base_network_stu(inputs_target)
    #
    #     '''
    #     DAN的MK-MMD Loss
    #     '''
    #     args.non_linear = False
    #     classifier_loss = criterion(outputs_source, labels_source)
    #     mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
    #         kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
    #         linear=not args.non_linear
    #     )
    #     alignment_loss = mkmmd_loss(features_source, features_target)
    #
    #     args.t_mcc = 2
    #     transfer_loss = ClassConfusionLoss(t=args.t_mcc)(outputs_target)
    #     classifier_loss = criterion(outputs_source, labels_source)
    #     total_loss2 = transfer_loss + classifier_loss + alignment_loss
    #
    #     optimizer_ff.zero_grad()
    #     optimizer_cc.zero_grad()
    #     total_loss2.backward()
    #     optimizer_ff.step()
    #     optimizer_cc.step()
    #
    #     if iter_num % interval_iter == 0 or iter_num == max_iter:
    #         base_network.eval()
    #
    #         acc_t_te, pre_t_te, rec_t_te, f1_t_te, _ = cal_acc_comb(dset_loaders["Target"], base_network_stu, args=args)
    #         log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%  Pre = {:.2f}%  Rec = {:.2f}%  F1 = {:.2f}%'. \
    #             format(args.task_str, int(iter_num // len(dset_loaders["source"])),
    #                    int(max_iter // len(dset_loaders["source"])), acc_t_te, pre_t_te, rec_t_te, f1_t_te)
    #         args.log.record(log_str)
    #         print(log_str)
    #         base_network.train()
    #
    print('Test Acc = {:.2f}%'.format(acc_t_te))
    print('Test Pre = {:.2f}%'.format(pre_t_te))
    print('Test Rec = {:.2f}%'.format(rec_t_te))
    print('Test F1 = {:.2f}%'.format(f1_t_te))

    gc.collect()
    torch.cuda.empty_cache()

    return acc_t_te, pre_t_te, rec_t_te, f1_t_te







    # '''
    # Contrastive Stage
    # '''
    # X_1, y_1, num_subjects_1, paradigm_1, sample_rate_1, ch_num_1 = data_process(args.data1) # (1296, 22, 1001)
    # X_2, y_2, num_subjects_2, paradigm_2, sample_rate_2, ch_num_2 = data_process(args.data2) # (6520, 3, 1126)
    #
    # if args.align:
    #     print("-------   START EA: -------")
    #     X_1 = data_alignment(X_1, args.N1, args.data1)
    #     X_2 = data_alignment(X_2, args.N2, args.data2)
    #     print("-------   FINISH EA: -------========")
    # X_1 = X_1[:, :, 0:1001]
    # X_2 = X_2[:, :, 0:1001]
    # dset_loaders = data_loader(X_1, y_1, X_2, y_2, args)
    #
    # '''
    # 001的(1296, 22, 1001)
    # '''
    # # args.feature_deep_dim = 4960
    # args.feature_deep_dim = 248
    # netF, netC = backbone_net(args, return_type='001xy')
    # if args.data_env != 'local':
    #     netF, netC = netF.cuda(), netC.cuda()
    # base_network_1 = nn.Sequential(netF, netC)
    # criterion = nn.CrossEntropyLoss()
    # optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
    # optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)
    #
    # '''
    # 004的(6520, 3, 1126)
    # '''
    # args.feature_deep_dim = 248
    # netF, netC = backbone_net(args, return_type='xy')
    # if args.data_env != 'local':
    #     netF, netC = netF.cuda(), netC.cuda()
    # base_network_2 = nn.Sequential(netF, netC)
    # optimizer_ff = optim.Adam(netF.parameters(), lr=args.lr)
    # optimizer_cc = optim.Adam(netC.parameters(), lr=args.lr)
    #
    # max_iter = args.max_epoch * len(dset_loaders["source"])
    # interval_iter = max_iter // args.max_epoch
    # args.max_iter = max_iter
    # iter_num = 0
    # base_network_1.train()
    # base_network_2.train()
    #
    # while iter_num < max_iter:
    #     iter_source = iter(dset_loaders["source"])
    #     inputs_source, labels_source = next(iter_source)  # torch.Size([32, 1, 22, 1001])
    #
    #     iter_target = iter(dset_loaders["target"])
    #     inputs_target, _ = next(iter_target)
    #
    #     iter_num += 1
    #     '''
    #     features_source:  torch.Size([32, 4960]), netF后加一个MLP转化为([32, 280])
    #     outputs_source:  torch.Size([32, 2])
    #     features_target:  torch.Size([32, 280])
    #     outputs_target:  torch.Size([32, 2])
    #     '''
    #     features_source, outputs_source = base_network_1(inputs_source)
    #     features_target, outputs_target = base_network_2(inputs_target)
    #
    #
    #     '''
    #     DAN的MK-MMD Loss
    #     '''
    #     args.non_linear = False
    #     args.alignment_weight = 1.0
    #     classifier_loss = criterion(outputs_source, labels_source)
    #     mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
    #         kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
    #         linear=not args.non_linear
    #     )
    #     alignment_loss = mkmmd_loss(features_source, features_target)
    #
    #     args.loss_trade_off = 1.0
    #     args.t_mcc = 2
    #     transfer_loss = ClassConfusionLoss(t=args.t_mcc)(outputs_target)
    #     classifier_loss = criterion(outputs_source, labels_source)
    #     # total_loss = args.loss_trade_off * transfer_loss + classifier_loss
    #     total_loss = args.loss_trade_off * transfer_loss + classifier_loss + alignment_loss
    #
    #     optimizer_f.zero_grad()
    #     optimizer_c.zero_grad()
    #     optimizer_ff.zero_grad()
    #     optimizer_cc.zero_grad()
    #     total_loss.backward()
    #     optimizer_f.step()
    #     optimizer_c.step()
    #     optimizer_ff.step()
    #     optimizer_cc.step()
    #
    #     if iter_num % interval_iter == 0 or iter_num == max_iter:
    #         base_network_2.eval()
    #
    #         acc_t_te, pre_t_te, rec_t_te, f1_t_te, _ = cal_acc_comb(dset_loaders["Target"], base_network_2, args=args)
    #         log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%  Pre = {:.2f}%  Rec = {:.2f}%  F1 = {:.2f}%'. \
    #                   format(args.task_str, int(iter_num // len(dset_loaders["source"])),
    #                   int(max_iter // len(dset_loaders["source"])), acc_t_te, pre_t_te, rec_t_te, f1_t_te)
    #         args.log.record(log_str)
    #         print(log_str)
    #         base_network_2.train()
    #
    # print('Test Acc = {:.2f}%'.format(acc_t_te))
    # print('Test Pre = {:.2f}%'.format(pre_t_te))
    # print('Test Rec = {:.2f}%'.format(rec_t_te))
    # print('Test F1 = {:.2f}%'.format(f1_t_te))
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # X_src, y_src, X_tar, y_tar = read_mi_combine_tar_diff(args)
    # print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    # dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)
    #
    # # netF, netC = backbone_net(args, return_type='xy')
    # # if args.data_env != 'local':
    # #     netF, netC = netF.cuda(), netC.cuda()
    # # base_network = nn.Sequential(netF, netC)
    # #
    # # criterion = nn.CrossEntropyLoss()
    # #
    # # optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
    # # optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)
    #
    # max_iter = args.max_epoch * len(dset_loaders["source"])
    # # max_iter = args.max_epoch * len(dset_loaders["source-Imbalanced"])
    # interval_iter = max_iter // args.max_epoch
    # args.max_iter = max_iter
    # iter_num = 0
    # base_network_2.train()
    #
    # while iter_num < max_iter:
    #     iter_source = iter(dset_loaders["source"])
    #     inputs_source, labels_source = next(iter_source)
    #
    #     iter_target = iter(dset_loaders["target"])
    #     inputs_target, _ = next(iter_target)
    #
    #     if inputs_source.size(0) == 1:
    #         continue
    #
    #     iter_num += 1
    #
    #     # print("TWO============================")
    #     # print(inputs_source.shape)
    #     # print(inputs_target.shape)
    #     features_source, outputs_source = base_network_2(inputs_source)
    #     features_target, outputs_target = base_network_2(inputs_target)
    #
    #     '''
    #     DAN的MK-MMD Loss
    #     '''
    #     args.non_linear = False
    #     args.alignment_weight = 1.0
    #     classifier_loss = criterion(outputs_source, labels_source)
    #     mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
    #         kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
    #         linear=not args.non_linear
    #     )
    #     alignment_loss = mkmmd_loss(features_source, features_target)
    #
    #     args.loss_trade_off = 1.0
    #     args.t_mcc = 2
    #     transfer_loss = ClassConfusionLoss(t=args.t_mcc)(outputs_target)
    #     classifier_loss = criterion(outputs_source, labels_source)
    #     # total_loss = args.loss_trade_off * transfer_loss + classifier_loss
    #     total_loss = args.loss_trade_off * transfer_loss + classifier_loss + alignment_loss
    #
    #     optimizer_ff.zero_grad()
    #     optimizer_cc.zero_grad()
    #     total_loss.backward()
    #     optimizer_ff.step()
    #     optimizer_cc.step()
    #
    #     if iter_num % interval_iter == 0 or iter_num == max_iter:
    #         base_network_2.eval()
    #
    #         acc_t_te, pre_t_te, rec_t_te, f1_t_te, _ = cal_acc_comb(dset_loaders["Target"], base_network_2, args=args)
    #         log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%  Pre = {:.2f}%  Rec = {:.2f}%  F1 = {:.2f}%'. \
    #                   format(args.task_str, int(iter_num // len(dset_loaders["source"])),
    #                   int(max_iter // len(dset_loaders["source"])), acc_t_te, pre_t_te, rec_t_te, f1_t_te)
    #         args.log.record(log_str)
    #         print(log_str)
    #         base_network_2.train()
    #
    # print('Test Acc = {:.2f}%'.format(acc_t_te))
    # print('Test Pre = {:.2f}%'.format(pre_t_te))
    # print('Test Rec = {:.2f}%'.format(rec_t_te))
    # print('Test F1 = {:.2f}%'.format(f1_t_te))
    #
    # gc.collect()
    # torch.cuda.empty_cache()
    #
    # return acc_t_te, pre_t_te, rec_t_te, f1_t_te


if __name__ == '__main__':

    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False

    dataset1 = 'BNCI2014001'
    dataset2 = 'BNCI2014004'
    # dataset1 = 'BNCI2014001_filter'
    # dataset2 = 'BNCI2014004_filter'

    dct = pd.DataFrame(
        columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                 's12', 's13'])

    paradigm1, N1, chn1, class_num1, time_sample_num1, sample_rate1 = 'MI', 9, 22, 2, 1001, 250
    paradigm2, N2, chn2, class_num2, time_sample_num2, sample_rate2 = 'MI', 9, 3, 2, 1126, 250

    # args = argparse.Namespace(feature_deep_dim=feature_deep_dim, trial_num=trial_num,
    #                           time_sample_num=time_sample_num, sample_rate=sample_rate,
    #                           N=N, chn=chn, class_num=class_num, paradigm=paradigm)

    args = argparse.Namespace(N1=N1, data_name1=dataset1,
                              N2=N2, data_name2=dataset2,
                              class_num=2, chn1=22, chn2=3, time_sample_num=1001, layer='wn',
                              sample_rate=250, feature_deep_dim=248)

    args.method = 'MCC'
    args.backbone = 'EEGNet'

    # whether to use EA
    args.align = True

    # learning rate
    args.lr = 0.001
    print("+_+_+_++_+_+_+_+_+_+_++_")
    print(f"{args.lr:.5f}")
    print("+_+_+_++_+_+_+_+_+_+_++_")

    # train batch size
    args.batch_size = 512

    # training epochs
    # args.max_epoch = 45

    # GPU device id
    try:
        device_id = str(sys.argv[1])
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
    except:
        args.data_env = 'local'

    total_acc = []
    total_pre = []
    total_rec = []
    total_f1 = []

    for s in [1, 2, 3, 4, 5]:
        args.SEED = s

        fix_random_seed(args.SEED)
        torch.backends.cudnn.deterministic = True

        args.data1 = dataset1
        args.data2 = dataset2
        print(args.data1)
        print(args.data2)
        print(args.method)
        print(args.SEED)
        print(args)
        print("--------------  start running:   --------------")

        args.local_dir = './data/' + str(dataset1) + '2' + str(dataset1) + '/'
        args.result_dir = './logs/'
        my_log = LogRecord(args)
        my_log.log_init()
        my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

        sub_acc_all = np.zeros(N2)
        sub_pre_all = np.zeros(N2)
        sub_rec_all = np.zeros(N2)
        sub_f1_all = np.zeros(N2)
        for idt in range(N2):
            args.idt = idt
            source_str = 'Except_S' + str(idt)
            target_str = 'S' + str(idt)
            args.task_str = source_str + '_2_' + target_str
            info_str = '\n========================== Transfer to ' + target_str + ' =========================='
            print(info_str)
            my_log.record(info_str)
            args.log = my_log

            sub_acc_all[idt], sub_pre_all[idt], sub_rec_all[idt], sub_f1_all[idt] = train_target(args)
            print('Sub acc: ', np.round(sub_acc_all, 3))
            print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
            print('Sub pre: ', np.round(sub_pre_all, 3))
            print('Avg pre: ', np.round(np.mean(sub_pre_all), 3))
            print('Sub rec: ', np.round(sub_rec_all, 3))
            print('Avg rec: ', np.round(np.mean(sub_rec_all), 3))
            print('Sub f1: ', np.round(sub_f1_all, 3))
            print('Avg f1: ', np.round(np.mean(sub_f1_all), 3))
            total_acc.append(sub_acc_all)
            total_pre.append(sub_pre_all)
            total_rec.append(sub_rec_all)
            total_f1.append(sub_f1_all)

            acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
            acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
            pre_sub_str = str(np.round(sub_pre_all, 3).tolist())
            pre_mean_str = str(np.round(np.mean(sub_pre_all), 3).tolist())
            rec_sub_str = str(np.round(sub_rec_all, 3).tolist())
            rec_mean_str = str(np.round(np.mean(sub_rec_all), 3).tolist())
            f1_sub_str = str(np.round(sub_f1_all, 3).tolist())
            f1_mean_str = str(np.round(np.mean(sub_f1_all), 3).tolist())
            args.log.record("\n==========================================")
            args.log.record(acc_sub_str)
            args.log.record(acc_mean_str)
            args.log.record(pre_sub_str)
            args.log.record(pre_mean_str)
            args.log.record(rec_sub_str)
            args.log.record(rec_mean_str)
            args.log.record(f1_sub_str)
            args.log.record(f1_mean_str)

        args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)

        print('total_acc is : ', str(total_acc))
        print('total_pre is : ', str(total_pre))
        print('total_rec is : ', str(total_rec))
        print('total_f1 is : ', str(total_f1))

        args.log.record(str(total_acc))
        args.log.record(str(total_pre))
        args.log.record(str(total_rec))
        args.log.record(str(total_f1))

        subject_acc_mean = np.round(np.average(total_acc, axis=0), 5)
        total_acc_mean = np.round(np.average(np.average(total_acc)), 5)
        total_acc_std = np.round(np.std(np.average(total_acc, axis=1)), 5)

        subject_pre_mean = np.round(np.average(total_pre, axis=0), 5)
        total_pre_mean = np.round(np.average(np.average(total_pre)), 5)
        total_pre_std = np.round(np.std(np.average(total_pre, axis=1)), 5)

        subject_rec_mean = np.round(np.average(total_rec, axis=0), 5)
        total_rec_mean = np.round(np.average(np.average(total_rec)), 5)
        total_rec_std = np.round(np.std(np.average(total_rec, axis=1)), 5)

        subject_f1_mean = np.round(np.average(total_f1, axis=0), 5)
        total_f1_mean = np.round(np.average(np.average(total_f1)), 5)
        total_f1_std = np.round(np.std(np.average(total_f1, axis=1)), 5)
        print("+-+-+-+-+_+_+_+__+_+_+++_++_+_+_+_++_++_+_+_+_+_+_+_+_+_+_+_+_+")

        print('subject_acc_mean is : ', subject_acc_mean)
        print('total_acc_mean is : ', total_acc_mean)
        print('total_acc_std is : ', total_acc_std)

        print('subject_pre_mean is : ', subject_pre_mean)
        print('total_pre_mean is : ', total_pre_mean)
        print('total_pre_std is : ', total_pre_std)

        print('subject_rec_mean is : ', subject_rec_mean)
        print('total_rec_mean is : ', total_rec_mean)
        print('total_rec_std is : ', total_rec_std)

        print('subject_f1_mean is : ', subject_f1_mean)
        print('total_f1_mean is : ', total_f1_mean)
        print('total_f1_std is : ', total_f1_std)

        args.log.record(str(subject_acc_mean))
        args.log.record(str(total_acc_mean))
        args.log.record(str(total_acc_std))

        args.log.record(str(subject_pre_mean))
        args.log.record(str(total_pre_mean))
        args.log.record(str(total_pre_std))

        args.log.record(str(subject_rec_mean))
        args.log.record(str(total_rec_mean))
        args.log.record(str(total_rec_std))

        args.log.record(str(subject_f1_mean))
        args.log.record(str(total_f1_mean))
        args.log.record(str(total_f1_std))

        source_str = str(dataset1)
        target_str = str(dataset2)
        data_name = source_str + ' 2 ' + target_str
        result_dct = {'dataset': data_name, 'acc_avg': total_acc_mean, 'acc_std': total_acc_std,
                      'pre_avg': total_pre_mean, 'pre_std': total_pre_std,
                      'rec_avg': total_rec_mean, 'rec_std': total_rec_std,
                      'f1_avg': total_f1_mean, 'f1_std': total_f1_std}
        for i in range(len(subject_acc_mean)):
            result_dct['s' + str(i)] = subject_acc_mean[i]

        dct = dct.append(result_dct, ignore_index=True)

    # save results to csv
    dct.to_csv('./logs/' + str(args.method) + ".csv")
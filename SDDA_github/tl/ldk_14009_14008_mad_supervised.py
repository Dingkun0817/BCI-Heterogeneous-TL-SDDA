'''
teacher 和 student 同时训练 [2024.04.18最佳版]
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

import gc
import sys
import math


def train_target(args):

    args.max_epoch = 50
    args.lr = 0.001

    X_1, y_1, num_subjects_1, paradigm_1, sample_rate_1, ch_num_1 = data_process(args.data1)  # (1296, 22, 1001)
    X_2, y_2, num_subjects_2, paradigm_2, sample_rate_2, ch_num_2 = data_process(args.data2)  # (6520, 3, 1126)

    if args.align:
        print("-------   START EA: -------")
        # X_1 = data_alignment_session(X_1, args.N1, args.data1)
        X_1 = data_alignment(X_1, args.N1, args.data1)
        X_2 = data_alignment(X_2, args.N2, args.data2)
        print("-------   FINISH EA: -------========")

    '''
    First Stage: Use Classifier Loss Train Teacher, only with src data
    '''
    pos_indices = np.where(y_1 == 1)[0]
    neg_indices = np.where(y_1 == 0)[0]
    selected_pos_indices = pos_indices
    '''
    src domain不能shuffle，因为有蒸馏
    '''
    # selected_neg_indices = neg_indices[:2880]
    selected_neg_indices = neg_indices
    selected_indices = np.concatenate((selected_pos_indices, selected_neg_indices))
    np.random.seed(42)
    np.random.shuffle(selected_indices)  # 打乱索引
    X_1 = X_1[selected_indices]
    y_1 = y_1[selected_indices]
    dset_loaders1 = data_loader(X_1, y_1, X_2, y_2, args)


    '''
    teacher
    '''
    args.feature_deep_dim = 48
    args.chn = 16
    netF, netC = backbone_net(args, return_type='xy')
    # args.feature_deep_dim = 4960
    # netF, netC = backbone_net(args, return_type='xy')
    if args.data_env != 'local':
        netF, netC = netF.cuda(), netC.cuda()
    base_network = nn.Sequential(netF, netC)
    optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
    optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)


    '''
    student
    '''
    args.feature_deep_dim = 48
    args.chn = 8
    netFF, netCC = backbone_net(args, return_type='xy')
    if args.data_env != 'local':
        netFF, netCC = netFF.cuda(), netCC.cuda()
    base_network_stu = nn.Sequential(netFF, netCC)
    # criterion = nn.CrossEntropyLoss()
    '''
    Imbalanced数据集的交叉熵要加权重
    '''
    num_class_1 = 700
    num_class_0 = 3500
    total_samples = num_class_0 + num_class_1
    weight_class_0 = num_class_1 / total_samples
    weight_class_1 = num_class_0 / total_samples
    weight = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float32)
    weight = weight.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight)


    optimizer_ff = optim.Adam(netFF.parameters(), lr=args.lr)
    optimizer_cc = optim.Adam(netCC.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(dset_loaders1["source"])
    interval_iter = max_iter // args.max_epoch
    args.max_iter = max_iter
    iter_num = 0
    base_network.train()
    base_network_stu.train()


    X_1 = X_1[:, 0:8, :]
    X_2 = X_2[:, :, :206]
    train_x = X_1
    train_y = y_1

    data_subjects = np.split(X_2, indices_or_sections=8, axis=0)
    labels_subjects = np.split(y_2, indices_or_sections=8, axis=0)
    X_2 = data_subjects.pop(args.idt)
    y_2 = labels_subjects.pop(args.idt)

    pos_indices = np.where(y_2 == 1)[0]
    neg_indices = np.where(y_2 == 0)[0]
    selected_pos_indices = pos_indices
    # selected_neg_indices = neg_indices[:700]
    selected_neg_indices = neg_indices
    selected_indices = np.concatenate((selected_pos_indices, selected_neg_indices))
    # np.random.shuffle(selected_indices)   # 打乱索引
    X_2 = X_2[selected_indices]
    y_2 = y_2[selected_indices]
    np.set_printoptions(threshold=np.inf)


    '''
    32  +  160
    '''
    indices_1 = np.arange(0, 32)
    indices_2 = np.arange(700, 700 + 160)
    indices = np.concatenate((indices_1, indices_2))
    test_x = X_2[indices, :, :]
    test_y = y_2[indices]
    # 打乱test_x和test_y
    shuffled_indices = np.random.permutation(len(test_x))
    test_x = test_x[shuffled_indices]
    test_y = test_y[shuffled_indices]
    # 剩余的样本  (前面都是很1，后面都是0，完全黑盒)
    remaining_indices = np.setdiff1d(np.arange(4200), indices)
    test_xx = X_2[remaining_indices, :, :]
    test_yy = y_2[remaining_indices]


    #
    # test_x = X_2[0:128, :, :]
    # test_y = y_2[0:128]
    # test_xx = X_2[128:4200, :, :]
    # test_yy = y_2[128:4200]

    print('train_x, train_y, test_x, test_y, test_xx, test_yy', train_x.shape, train_y.shape, test_x.shape, test_y.shape, test_xx.shape, test_yy.shape)
    print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    dset_loaders2 = data_loader(train_x, train_y, test_x, test_y, args)
    dset_loaders3 = data_loader(train_x, train_y, test_xx, test_yy, args)

    while iter_num < max_iter:
        try:
            inputs_source111, labels_source111 = next(iter_source)
            inputs_source, labels_source = next(iter_source2)
        except:
            iter_source = iter(dset_loaders1["source"])
            inputs_source111, labels_source111 = next(iter_source)
            iter_source2 = iter(dset_loaders2["source"])
            inputs_source, labels_source = next(iter_source2)
        try:
            inputs_target, labels_target = next(iter_target)
        except:
            iter_target = iter(dset_loaders2["target"])
            inputs_target, labels_target = next(iter_target)

        try:
            inputs_target_unlabel, _ = next(iter_target_unlabel)
        except:
            iter_target_unlabel = iter(dset_loaders3["target"])
            inputs_target_unlabel, _ = next(iter_target_unlabel)

        iter_num += 1

        teacher_source, outputs_source_teacher = base_network(inputs_source111)

        args.non_linear = False
        args.alignment_weight = 1.0
        classifier_loss1 = criterion(outputs_source_teacher, labels_source111)

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        classifier_loss1.backward()
        optimizer_f.step()
        optimizer_c.step()

        teacher_source = teacher_source.detach()
        outputs_source_teacher = outputs_source_teacher.detach()


        features_source, outputs_source = base_network_stu(inputs_source)
        features_target, outputs_target = base_network_stu(inputs_target)

        args.non_linear = True
        classifier_loss_src = criterion(outputs_source, labels_source)
        classifier_loss_tar = criterion(outputs_target, labels_target)

        mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
            linear=not args.non_linear
        )
        alignment_loss2 = mkmmd_loss(features_target, features_source)


        def distillation(y, labels, teacher_scores, temp, alpha):
            return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
                    temp * temp * 2.0 * alpha)

        ditillation_loss = distillation(outputs_source, labels_source, outputs_source_teacher, temp=8, alpha=0.5)   # 5  0.7

        '''
        MCC最小类混淆Loss
        '''
        args.t_mcc = 3
        transfer_loss = ClassConfusionLoss(t=args.t_mcc)(outputs_target)

        # total_loss = classifier_loss_src + classifier_loss_tar + ditillation_loss + alignment_loss2 + 1.0 * transfer_loss   # 2
        total_loss = classifier_loss_src + classifier_loss_tar + 1.0 * ditillation_loss

        optimizer_ff.zero_grad()
        optimizer_cc.zero_grad()
        total_loss.backward()
        optimizer_ff.step()
        optimizer_cc.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network_stu.eval()

            # acc_t_te, pre_t_te, rec_t_te, f1_t_te, _ = cal_acc_comb(dset_loaders3["Target"], base_network_stu, args=args)
            auc_t_te, _ = cal_auc_comb(dset_loaders3["Target"], base_network_stu, args=args)
            # log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%  Pre = {:.2f}%  Rec = {:.2f}%  F1 = {:.2f}%'. \
            #     format(args.task_str, int(iter_num // len(dset_loaders2["source"])),
            #            int(max_iter // len(dset_loaders2["source"])), acc_t_te, pre_t_te, rec_t_te, f1_t_te)
            log_str = 'Task: {}, Iter:{}/{}; Auc = {:.2f}% '. \
                format(args.task_str, int(iter_num // len(dset_loaders2["source"])),
                       int(max_iter // len(dset_loaders2["source"])), auc_t_te)
            args.log.record(log_str)
            print(log_str)
            base_network_stu.train()

    print('Test Auc = {:.2f}%'.format(auc_t_te))
    # print('Test Acc = {:.2f}%'.format(acc_t_te))
    # print('Test Pre = {:.2f}%'.format(pre_t_te))
    # print('Test Rec = {:.2f}%'.format(rec_t_te))
    # print('Test F1 = {:.2f}%'.format(f1_t_te))

    gc.collect()
    torch.cuda.empty_cache()

    return auc_t_te
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

    dataset1 = 'BNCI2014009'
    dataset2 = 'BNCI2014008'

    dct = pd.DataFrame(
        columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                 's12', 's13'])

    paradigm1, N1, chn1, class_num1, time_sample_num1, sample_rate1 = 'MI', 10, 16, 2, 206, 256
    paradigm2, N2, chn2, class_num2, time_sample_num2, sample_rate2 = 'MI', 8, 8, 2, 257, 256

    args = argparse.Namespace(N1=N1, data_name1=dataset1,
                              N2=N2, data_name2=dataset2,
                              class_num=2, chn1=10, chn2=8, time_sample_num=206, layer='wn',
                              sample_rate=256, feature_deep_dim=288)

    args.method = 'MAD'
    args.backbone = 'EEGNet'

    # whether to use EA
    args.align = True

    # learning rate
    args.lr = 0.001

    # train batch size
    args.batch_size = 32

    # training epochs
    # args.max_epoch = 45

    # GPU device id
    try:
        device_id = str(sys.argv[1])
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
    except:
        args.data_env = 'local'

    total_auc = []

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

        sub_auc_all = np.zeros(N2)
        # sub_acc_all = np.zeros(N2)
        # sub_pre_all = np.zeros(N2)
        # sub_rec_all = np.zeros(N2)
        # sub_f1_all = np.zeros(N2)
        for idt in range(N2):  # 留一法, idt作为test, 非idt作为train
            args.idt = idt
            source_str = str(dataset1)
            target_str = str(dataset2) + '    subject ' + str(idt)
            args.task_str = source_str + '_2_' + target_str
            info_str = '\n========================== Transfer to ' + target_str + ' =========================='
            print(info_str)
            my_log.record(info_str)
            args.log = my_log

            # sub_acc_all[idt], sub_pre_all[idt], sub_rec_all[idt], sub_f1_all[idt] = train_target(args)
            sub_auc_all[idt] = train_target(args)
        print('Sub auc: ', np.round(sub_auc_all, 3))
        print('Avg auc: ', np.round(np.mean(sub_auc_all), 3))

        total_auc.append(sub_auc_all)

        auc_sub_str = str(np.round(sub_auc_all, 3).tolist())
        auc_mean_str = str(np.round(np.mean(sub_auc_all), 3).tolist())

        args.log.record("\n==========================================")
        args.log.record(auc_sub_str)
        args.log.record(auc_mean_str)

    args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)

    print('total_auc is : ', str(total_auc))

    args.log.record(str(total_auc))

    subject_auc_mean = np.round(np.average(total_auc, axis=0), 5)
    total_auc_mean = np.round(np.average(np.average(total_auc)), 5)
    total_auc_std = np.round(np.std(np.average(total_auc, axis=1)), 5)

    print("+-+-+-+-+_+_+_+__+_+_+++_++_+_+_+_++_++_+_+_+_+_+_+_+_+_+_+_+_+")

    print('subject_auc_mean is : ', subject_auc_mean)
    print('total_auc_mean is : ', total_auc_mean)
    print('total_auc_std is : ', total_auc_std)

    args.log.record(str(subject_auc_mean))
    args.log.record(str(total_auc_mean))
    args.log.record(str(total_auc_std))

    source_str = str(dataset1)
    target_str = str(dataset2)
    dataset = source_str + ' 2 ' + target_str
    result_dct = {'dataset': dataset, 'auc_avg': total_auc_mean, 'auc_std': total_auc_std}

    for i in range(len(subject_auc_mean)):
        result_dct['s' + str(i)] = subject_auc_mean[i]

    dct = dct.append(result_dct, ignore_index=True)

    # save results to csv
    dct.to_csv('./logs/' + str(args.method) + ".csv")
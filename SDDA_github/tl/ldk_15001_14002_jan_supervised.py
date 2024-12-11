import sys
# sys.path.append('E:\Pycharm_BCI\Self_Study\T-TIME')
sys.path.append('/data1/ldk/T-TIME')
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
from utils.dataloader import read_mi_combine_tar, read_mi_combine_tar_diff, read_mi_combine_tar_diff_supervised_2
from utils.utils_1 import lr_scheduler_full, fix_random_seed, cal_acc_comb, data_loader
from utils.loss import JointMultipleKernelMaximumMeanDiscrepancy, GaussianKernel
from torch.nn.functional import softmax
from tl.utils.utils_1 import data_alignment, data_alignment_session
from utils.dataloader import data_process

import gc
import sys


def train_target(args):
    X_src, y_src, X_tar, y_tar, X_tar_unlabel, y_tar_unlabel = read_mi_combine_tar_diff_supervised_2(args)
    print('X_src, y_src, X_tar, y_tar, X_tar_unlabel, y_tar_unlabel:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape, X_tar_unlabel.shape, y_tar_unlabel.shape)
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)
    dset_loaders2 = data_loader(X_src, y_src, X_tar_unlabel, y_tar_unlabel, args)

    print('======================subject {}============================='.format(args.idt))
    args.chn = 3
    netF, netC = backbone_net(args, return_type='xy')
    if args.data_env != 'local':
        netF, netC = netF.to(device), netC.to(device)
    base_network = nn.Sequential(netF, netC)

    criterion = nn.CrossEntropyLoss()

    optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
    optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(dset_loaders["source"])
    interval_iter = max_iter // args.max_epoch
    args.max_iter = max_iter
    iter_num = 0
    base_network.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source = next(iter_source)
        try:
            inputs_target, labels_target = next(iter_target)
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_target, labels_target = next(iter_target)
        try:
            inputs_target_unlabel, _ = next(iter_target_unlabel)
        except:
            iter_target_unlabel = iter(dset_loaders2["target"])
            inputs_target_unlabel, _ = next(iter_target_unlabel)

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1

        inputs_source, inputs_target, labels_source = inputs_source.to(device), inputs_target.to(device), labels_source.to(device)
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)

        classifier_loss_src = criterion(outputs_source, labels_source)
        classifier_loss_tar = criterion(outputs_target, labels_target)
        args.alignment_weight = 1.0
        args.linear = False
        thetas = None
        jmmd_loss = JointMultipleKernelMaximumMeanDiscrepancy(
            kernels=(
                [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
                (GaussianKernel(sigma=0.92, track_running_stats=False),)
            ),
            linear=args.linear, thetas=thetas
        )
        alignment_loss = jmmd_loss(
            (features_source, softmax(outputs_source, dim=1)),
            (features_target, softmax(outputs_target, dim=1))
        )
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print(features_source.shape)
        # print(outputs_source.shape)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        total_loss = classifier_loss_src + classifier_loss_tar + alignment_loss * args.alignment_weight

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        total_loss.backward()
        optimizer_f.step()
        optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()

            acc_t_te, pre_t_te, rec_t_te, f1_t_te, _ = cal_acc_comb(dset_loaders2["Target"], base_network, args=args)
            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%  Pre = {:.2f}%  Rec = {:.2f}%  F1 = {:.2f}%'. \
                       format(args.task_str, int(iter_num // len(dset_loaders["source"])),
                       int(max_iter // len(dset_loaders["source"])), acc_t_te, pre_t_te, rec_t_te, f1_t_te)
            args.log.record(log_str)
            print(log_str)

            base_network.train()

    print('Test Acc = {:.2f}%'.format(acc_t_te))
    print('Test Pre = {:.2f}%'.format(pre_t_te))
    print('Test Rec = {:.2f}%'.format(rec_t_te))
    print('Test F1 = {:.2f}%'.format(f1_t_te))

    gc.collect()
    torch.cuda.empty_cache()

    return acc_t_te, pre_t_te, rec_t_te, f1_t_te


if __name__ == '__main__':

    start_time = time.time()
    dataset1 = 'BNCI2015001'
    dataset2 = 'BNCI2014002'

    dct = pd.DataFrame(
        columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                 's12', 's13'])

    paradigm1, N1, chn1, class_num1, time_sample_num1, sample_rate1 = 'MI', 12, 13, 2, 2561, 512
    paradigm2, N2, chn2, class_num2, time_sample_num2, sample_rate2 = 'MI', 14, 15, 2, 2561, 512

    args = argparse.Namespace(N1=N1, data_name1=dataset1,
                              N2=N2, data_name2=dataset2,
                              class_num=2, chn1=13, chn2=15, time_sample_num=2561, layer='wn',
                              sample_rate=512, feature_deep_dim=640)

    args.method = 'JAN'
    args.backbone = 'EEGNet'

    # whether to use EA
    args.align = True

    # learning rate
    args.lr = 0.001

    # train batch size
    args.batch_size = 32

    # training epochs
    args.max_epoch = 100

    # GPU device id
    try:
        device_id = str(sys.argv[1])
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
    except:
        args.data_env = 'local'
    device = torch.device("cuda" if args.data_env == 'gpu' else "cpu")

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
                                        'f1_avg': total_f1_mean,   'f1_std': total_f1_std}



    for i in range(len(subject_acc_mean)):
        result_dct['s' + str(i)] = subject_acc_mean[i]

    dct = dct.append(result_dct, ignore_index=True)

    # save results to csv
    dct.to_csv('./logs/' + str(args.method) + ".csv")

    '''
    计算代码运行时间
    '''
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"代码运行时间：{elapsed_time} 秒")
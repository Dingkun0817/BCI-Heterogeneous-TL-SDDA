'''
这个代码中以cdan为例尝试了auto-correlation
'''

import sys
# sys.path.append('E:\Pycharm_BCI\Self_Study\T-TIME')
sys.path.append('/data1/ldk/T-TIME')

import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from utils.network import backbone_net, AdversarialNetwork
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar, read_mi_combine_tar_diff, read_mi_combine_tar_diff_supervised_3
from utils.utils_1 import lr_scheduler_full, fix_random_seed, cal_acc_comb, data_loader, cal_auc_comb
from utils.loss import CDANE, Entropy, RandomLayer
from utils.network import calc_coeff
from utils.auto_correlation import operation_A
import matplotlib.pyplot as plt
from tl.utils.utils_1 import data_alignment, data_alignment_session
from utils.dataloader import data_process

import gc
import sys


def train_target(args):
    X_src, y_src, X_tar, y_tar, X_tar_unlabel, y_tar_unlabel = read_mi_combine_tar_diff_supervised_3(args)
    print('X_src, y_src, X_tar, y_tar, X_tar_unlabel, y_tar_unlabel:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape, X_tar_unlabel.shape, y_tar_unlabel.shape)
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)
    dset_loaders2 = data_loader(X_src, y_src, X_tar_unlabel, y_tar_unlabel, args)

    print('======================subject {}============================='.format(args.idt))

    args.chn = 8
    netF, netC = backbone_net(args, return_type='xy')
    if args.data_env != 'local':
        # netF, netC = netF.cuda(), netC.cuda()
        netF.cuda(), netC.cuda()
    base_network = nn.Sequential(netF, netC)

    args.max_iter = args.max_epoch * len(dset_loaders["source"])

    ad_net = AdversarialNetwork(args.feature_deep_dim * args.class_num, 32, 8)
    # ad_net = AdversarialNetwork(args.feature_deep_dim, 32, 8)   # 走random_layer
    if args.data_env != 'local':
        # ad_net = ad_net.cuda()
        ad_net.cuda()
    random_layer = RandomLayer([args.feature_deep_dim, args.class_num], args.feature_deep_dim, use_cuda=args.data_env!='local')
    # if args.data_env != 'local':
        # random_layer = random_layer.cuda()
        # random_layer.cuda()

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

    optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
    optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)
    optimizer_d = optim.Adam(ad_net.parameters(), lr=args.lr)

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
        if args.data_env != 'local':
            # inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
            inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)   # 两个torch.Size([32, 128])拼成torch.Size([64, 128])

        args.loss_trade_off = 1.0
        outputs = torch.cat((outputs_source, outputs_target), dim=0)        #  torch.Size([64, 2])
        softmax_out = nn.Softmax(dim=1)(outputs)                            #  torch.Size([64, 2])

        entropy = Entropy(softmax_out)

        transfer_loss = CDANE([features, softmax_out], ad_net, random_layer, entropy, calc_coeff(iter_num), args)
        classifier_loss_src = criterion(outputs_source, labels_source)
        classifier_loss_tar = criterion(outputs_target, labels_target)
        total_loss = args.loss_trade_off * transfer_loss + classifier_loss_src + classifier_loss_tar

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        optimizer_d.zero_grad()
        total_loss.backward()
        optimizer_f.step()
        optimizer_c.step()
        optimizer_d.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()

            # acc_t_te, pre_t_te, rec_t_te, f1_t_te, _ = cal_acc_comb(dset_loaders2["Target"], base_network, args=args)
            auc_t_te, _ = cal_auc_comb(dset_loaders2["Target"], base_network, args=args)
            # log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%  Pre = {:.2f}%  Rec = {:.2f}%  F1 = {:.2f}%'. \
            #            format(args.task_str, int(iter_num // len(dset_loaders["source"])),
            #            int(max_iter // len(dset_loaders["source"])), acc_t_te, pre_t_te, rec_t_te, f1_t_te)
            log_str = 'Task: {}, Iter:{}/{}; Auc = {:.2f}% '. \
                format(args.task_str, int(iter_num // len(dset_loaders["source"])),
                       int(max_iter // len(dset_loaders["source"])), auc_t_te)
            args.log.record(log_str)
            print(log_str)

            base_network.train()

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
                              sample_rate=256, feature_deep_dim=48)

    args.method = 'CDAN'
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
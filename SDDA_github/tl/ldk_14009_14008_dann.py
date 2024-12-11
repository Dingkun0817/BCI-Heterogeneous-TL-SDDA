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
from utils.network import backbone_net, feat_classifier
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar, read_mi_combine_tar_diff
from utils.utils_1 import lr_scheduler_full, fix_random_seed, cal_acc_comb, data_loader, cal_auc_comb
from utils.loss import CELabelSmooth_raw, Entropy, ReverseLayerF
from tl.utils.utils_1 import data_alignment, data_alignment_session
from utils.dataloader import data_process

import gc
import sys


def train_target(args):
    X_1, y_1, num_subjects_1, paradigm_1, sample_rate_1, ch_num_1 = data_process(args.data1)
    X_2, y_2, num_subjects_2, paradigm_2, sample_rate_2, ch_num_2 = data_process(args.data2)
    if args.align:
        print("-------   START EA: -------")
        # X_1 = data_alignment_session(X_1, args.N1, args.data1)
        X_1 = data_alignment(X_1, args.N1, args.data1)
        X_2 = data_alignment(X_2, args.N2, args.data2)
        print("-------   FINISH EA: -------")
    X_1 = X_1[:, 0:8, :]
    X_2 = X_2[:, :, 0:206]

    print('======================subject {}============================='.format(args.idt))
    train_x = X_1
    train_y = y_1
    data_subjects = np.split(X_2, indices_or_sections=8, axis=0)
    labels_subjects = np.split(y_2, indices_or_sections=8, axis=0)
    test_x = data_subjects.pop(args.idt)
    test_y = labels_subjects.pop(args.idt)


    pos_indices = np.where(y_1 == 1)[0]
    neg_indices = np.where(y_1 == 0)[0]
    selected_pos_indices = pos_indices
    # selected_neg_indices = neg_indices[:2880]
    selected_neg_indices = neg_indices
    selected_indices = np.concatenate((selected_pos_indices, selected_neg_indices))
    X_1 = X_1[selected_indices]
    y_1 = y_1[selected_indices]
    train_x = X_1
    train_y = y_1

    pos_indices = np.where(test_y == 1)[0]
    neg_indices = np.where(test_y == 0)[0]
    selected_pos_indices = pos_indices
    # selected_neg_indices = neg_indices[:700]
    selected_neg_indices = neg_indices
    selected_indices = np.concatenate((selected_pos_indices, selected_neg_indices))
    test_x = test_x[selected_indices]
    test_y = test_y[selected_indices]


    print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    dset_loaders = data_loader(train_x, train_y, test_x, test_y, args)

    args.chn = 8
    netF, netC = backbone_net(args, return_type='xy')
    # if args.data_env != 'local':
    netF, netC = netF.to(device), netC.to(device)
    base_network = nn.Sequential(netF, netC)

    args.max_iter = args.max_epoch * len(dset_loaders["source"])

    ad_net = feat_classifier(type=args.layer, class_num=args.class_num, hidden_dim=args.feature_deep_dim).to(device)

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
        iter_source = iter(dset_loaders["source"])
        inputs_source, labels_source = next(iter_source)
        iter_target = iter(dset_loaders["target"])
        inputs_target, _ = next(iter_target)

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1

        inputs_source, inputs_target, labels_source = inputs_source.to(device), inputs_target.to(device), labels_source.to(device)
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)

        p = float(iter_num) / max_iter
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        reverse_source, reverse_target = ReverseLayerF.apply(features_source, alpha), ReverseLayerF.apply(features_target, alpha)
        domain_output_s = ad_net(reverse_source)
        domain_output_t = ad_net(reverse_target)
        domain_label_s = torch.ones(inputs_source.size()[0]).long().to(device)
        domain_label_t = torch.zeros(inputs_target.size()[0]).long().to(device)

        classifier_loss = criterion(outputs_source, labels_source)
        adv_loss = criterion(domain_output_s, domain_label_s) + criterion(domain_output_t, domain_label_t)
        total_loss = classifier_loss + adv_loss

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        optimizer_d.zero_grad()
        total_loss.backward()
        optimizer_f.step()
        optimizer_c.step()
        optimizer_d.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()

            # acc_t_te, pre_t_te, rec_t_te, f1_t_te, _ = cal_acc_comb(dset_loaders["Target"], base_network, args=args)
            # log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%  Pre = {:.2f}%  Rec = {:.2f}%  F1 = {:.2f}%'. \
            #     format(args.task_str, int(iter_num // len(dset_loaders["source"])),
            #            int(max_iter // len(dset_loaders["source"])), acc_t_te, pre_t_te, rec_t_te, f1_t_te)
            auc_t_te, _ = cal_auc_comb(dset_loaders["Target"], base_network, args=args)
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

    args.method = 'DANN'
    args.backbone = 'EEGNet'

    # args.alignment_weight = 1.0

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

    total_auc = []
    # total_acc = []
    # total_pre = []
    # total_rec = []
    # total_f1 = []

    for s in [1, 2, 3, 4, 5]:
    # for s in [1]:
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
        # print('Sub acc: ', np.round(sub_acc_all, 3))
        # print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
        # print('Sub pre: ', np.round(sub_pre_all, 3))
        # print('Avg pre: ', np.round(np.mean(sub_pre_all), 3))
        # print('Sub rec: ', np.round(sub_rec_all, 3))
        # print('Avg rec: ', np.round(np.mean(sub_rec_all), 3))
        # print('Sub f1: ', np.round(sub_f1_all, 3))
        # print('Avg f1: ', np.round(np.mean(sub_f1_all), 3))
        total_auc.append(sub_auc_all)
        # total_acc.append(sub_acc_all)
        # total_pre.append(sub_pre_all)
        # total_rec.append(sub_rec_all)
        # total_f1.append(sub_f1_all)

        auc_sub_str = str(np.round(sub_auc_all, 3).tolist())
        auc_mean_str = str(np.round(np.mean(sub_auc_all), 3).tolist())
        # acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
        # acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
        # pre_sub_str = str(np.round(sub_pre_all, 3).tolist())
        # pre_mean_str = str(np.round(np.mean(sub_pre_all), 3).tolist())
        # rec_sub_str = str(np.round(sub_rec_all, 3).tolist())
        # rec_mean_str = str(np.round(np.mean(sub_rec_all), 3).tolist())
        # f1_sub_str = str(np.round(sub_f1_all, 3).tolist())
        # f1_mean_str = str(np.round(np.mean(sub_f1_all), 3).tolist())
        args.log.record("\n==========================================")
        args.log.record(auc_sub_str)
        args.log.record(auc_mean_str)
        # args.log.record(acc_sub_str)
        # args.log.record(acc_mean_str)
        # args.log.record(pre_sub_str)
        # args.log.record(pre_mean_str)
        # args.log.record(rec_sub_str)
        # args.log.record(rec_mean_str)
        # args.log.record(f1_sub_str)
        # args.log.record(f1_mean_str)

    args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)

    print('total_auc is : ', str(total_auc))
    # print('total_acc is : ', str(total_acc))
    # print('total_pre is : ', str(total_pre))
    # print('total_rec is : ', str(total_rec))
    # print('total_f1 is : ', str(total_f1))

    args.log.record(str(total_auc))
    # args.log.record(str(total_acc))
    # args.log.record(str(total_pre))
    # args.log.record(str(total_rec))
    # args.log.record(str(total_f1))

    subject_auc_mean = np.round(np.average(total_auc, axis=0), 5)
    total_auc_mean = np.round(np.average(np.average(total_auc)), 5)
    total_auc_std = np.round(np.std(np.average(total_auc, axis=1)), 5)

    # subject_acc_mean = np.round(np.average(total_acc, axis=0), 5)
    # total_acc_mean = np.round(np.average(np.average(total_acc)), 5)
    # total_acc_std = np.round(np.std(np.average(total_acc, axis=1)), 5)
    #
    # subject_pre_mean = np.round(np.average(total_pre, axis=0), 5)
    # total_pre_mean = np.round(np.average(np.average(total_pre)), 5)
    # total_pre_std = np.round(np.std(np.average(total_pre, axis=1)), 5)
    #
    # subject_rec_mean = np.round(np.average(total_rec, axis=0), 5)
    # total_rec_mean = np.round(np.average(np.average(total_rec)), 5)
    # total_rec_std = np.round(np.std(np.average(total_rec, axis=1)), 5)
    #
    # subject_f1_mean = np.round(np.average(total_f1, axis=0), 5)
    # total_f1_mean = np.round(np.average(np.average(total_f1)), 5)
    # total_f1_std = np.round(np.std(np.average(total_f1, axis=1)), 5)
    print("+-+-+-+-+_+_+_+__+_+_+++_++_+_+_+_++_++_+_+_+_+_+_+_+_+_+_+_+_+")

    print('subject_auc_mean is : ', subject_auc_mean)
    print('total_auc_mean is : ', total_auc_mean)
    print('total_auc_std is : ', total_auc_std)

    # print('subject_acc_mean is : ', subject_acc_mean)
    # print('total_acc_mean is : ', total_acc_mean)
    # print('total_acc_std is : ', total_acc_std)
    #
    # print('subject_pre_mean is : ', subject_pre_mean)
    # print('total_pre_mean is : ', total_pre_mean)
    # print('total_pre_std is : ', total_pre_std)
    #
    # print('subject_rec_mean is : ', subject_rec_mean)
    # print('total_rec_mean is : ', total_rec_mean)
    # print('total_rec_std is : ', total_rec_std)
    #
    # print('subject_f1_mean is : ', subject_f1_mean)
    # print('total_f1_mean is : ', total_f1_mean)
    # print('total_f1_std is : ', total_f1_std)

    args.log.record(str(subject_auc_mean))
    args.log.record(str(total_auc_mean))
    args.log.record(str(total_auc_std))

    # args.log.record(str(subject_acc_mean))
    # args.log.record(str(total_acc_mean))
    # args.log.record(str(total_acc_std))
    #
    # args.log.record(str(subject_pre_mean))
    # args.log.record(str(total_pre_mean))
    # args.log.record(str(total_pre_std))
    #
    # args.log.record(str(subject_rec_mean))
    # args.log.record(str(total_rec_mean))
    # args.log.record(str(total_rec_std))
    #
    # args.log.record(str(subject_f1_mean))
    # args.log.record(str(total_f1_mean))
    # args.log.record(str(total_f1_std))

    source_str = str(dataset1)
    target_str = str(dataset2)
    dataset = source_str + ' 2 ' + target_str
    result_dct = {'dataset': dataset, 'auc_avg': total_auc_mean, 'auc_std': total_auc_std}
    # result_dct = {'dataset': dataset, 'acc_avg': total_acc_mean, 'acc_std': total_acc_std,
    #                                   'pre_avg': total_pre_mean, 'pre_std': total_pre_std,
    #                                   'rec_avg': total_rec_mean, 'rec_std': total_rec_std,
    #                                   'f1_avg': total_f1_mean, 'f1_std': total_f1_std}
    for i in range(len(subject_auc_mean)):
        result_dct['s' + str(i)] = subject_auc_mean[i]
    # for i in range(len(subject_acc_mean)):
    #     result_dct['s' + str(i)] = subject_acc_mean[i]

    dct = dct.append(result_dct, ignore_index=True)

    # save results to csv
    dct.to_csv('./logs/' + str(args.method) + ".csv")
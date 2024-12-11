import sys
# sys.path.append('E:\Pycharm_BCI\Self_Study\T-TIME')
sys.path.append('/data1/ldk/T-TIME')

import csv

import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from scipy.spatial.distance import cdist
import torch.nn.functional as F
from utils.network import backbone_net
from utils.loss import Entropy
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar, read_mi_combine_tar_diff, read_mi_combine_tar_diff_supervised_2
from utils.utils_1 import lr_scheduler_full, fix_random_seed, cal_acc_comb, data_loader
from utils.utils_1 import lr_scheduler, fix_random_seed, op_copy, cal_acc, cal_bca, cal_auc
from tl.utils.utils_1 import data_alignment, data_alignment_session
from utils.dataloader import data_process

import gc
import torch
import sys


def obtain_label(loader, netF, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netF(inputs)
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>args.threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return predict.astype('int')


def train_target(args):
    X_src, y_src, X_tar, y_tar, X_tar_unlabel, y_tar_unlabel = read_mi_combine_tar_diff_supervised_2(args)
    print('X_src, y_src, X_tar, y_tar, X_tar_unlabel, y_tar_unlabel:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape, X_tar_unlabel.shape, y_tar_unlabel.shape)
    dset_loaders = data_loader(X_src, y_src, X_tar, y_tar, args)
    dset_loaders2 = data_loader(X_src, y_src, X_tar_unlabel, y_tar_unlabel, args)

    print('======================subject {}============================='.format(args.idt))
    args.chn = 3
    netF, netC = backbone_net(args, return_type='y')
    if args.data_env != 'local':
        netF, netC = netF.cuda(), netC.cuda()
    base_network = nn.Sequential(netF, netC)

    criterion = nn.CrossEntropyLoss()

    ######################################################################################################
    # Source Model Training
    ######################################################################################################

    optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
    optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)

    args.batch_size = 32
    # TODO load pretrained model
    args.max_epoch = 100

    if not args.align:
        extra_string = '_noEA'
    else:
        extra_string = ''
    if args.max_epoch == 0:
        if args.align:    # ?????????????????????????????????????
            if args.data_env != 'local':
                base_network.load_state_dict(torch.load('./runs/' + str(args.data1) + '/' + str(args.backbone) +
                                                        '_S' + str(args.idt) + '_seed' + str(
                    args.SEED) + extra_string + '.ckpt'))
            else:
                base_network.load_state_dict(torch.load('./runs/' + str(args.data1) + '/' + str(args.backbone) +
                                                        '_S' + str(args.idt) + '_seed' + str(
                    args.SEED) + extra_string + '.ckpt', map_location=torch.device('cpu')))

    else:
        max_iter = args.max_epoch * len(dset_loaders["source"])
        #max_iter = args.max_epoch * len(dset_loaders["source-Imbalanced"])
        interval_iter = max_iter // args.max_epoch
        args.max_iter = max_iter
        iter_num = 0
        base_network.train()

        print('---------------Source Model Training---------------')

        while iter_num < max_iter:
            try:
                inputs_source, labels_source = next(iter_source)
            except:
                iter_source = iter(dset_loaders["source"])
                #iter_source = iter(dset_loaders["source-Imbalanced"])
                inputs_source, labels_source = next(iter_source)

            if inputs_source.size(0) == 1:
                continue

            iter_num += 1

            outputs_source = base_network(inputs_source)

            outputs_source = torch.nn.Softmax(dim=1)(outputs_source / 2)   # mcc里的温度系数

            args.trade_off = 1.0
            classifier_loss = criterion(outputs_source, labels_source)
            total_loss = classifier_loss

            optimizer_f.zero_grad()
            optimizer_c.zero_grad()
            total_loss.backward()
            optimizer_f.step()
            optimizer_c.step()

            if iter_num % interval_iter == 0 or iter_num == max_iter:
                base_network.eval()

                acc_t_te, pre_t_te, rec_t_te, f1_t_te, _ = cal_acc(dset_loaders2["Target"], netF, netC, args=args)
                log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%  Pre = {:.2f}%  Rec = {:.2f}%  F1 = {:.2f}%'. \
                           format(args.task_str, int(iter_num // len(dset_loaders["source"])),
                           int(max_iter // len(dset_loaders["source"])), acc_t_te, pre_t_te, rec_t_te, f1_t_te)
                args.log.record(log_str)
                print(log_str)

                base_network.train()

    #############################################################################################
    # Source HypOthesis Transfer
    ######################################################################################################

    print('+++++++++++++++++++++++Source HypOthesis Transfer+++++++++++++++++++++++')

    args.batch_size = 32
    args.max_epoch = 5
    #dset_loaders = data_loader(X_src, y_src, X_tar, y_tar#########, args)

    netC.eval()
    netF.train()

    optimizer = optim.Adam(netF.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    #max_iter = args.max_epoch * len(dset_loaders["target-Imbalanced"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_test, _ = next(iter_test)
            tar_id += 1
            tar_idx = np.arange(args.batch_size, dtype=int) + args.batch_size * tar_id
        except:
            iter_test = iter(dset_loaders["target"])
            #iter_test = iter(dset_loaders["target-Imbalanced"])
            inputs_test, _ = next(iter_test)
            tar_id = 0
            tar_idx = np.arange(args.batch_size, dtype=int)

        if inputs_test.size(0) == 1:
            continue

        if args.data_env != 'local':
            inputs_test = inputs_test.cuda()

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            mem_label = obtain_label(dset_loaders["Target"], netF, netC, args)
            mem_label = torch.from_numpy(mem_label)
            if args.data_env != 'local':
                mem_label = mem_label.cuda()
            netF.train()

        iter_num += 1
        #lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        features_test = netF(inputs_test)
        outputs_test = netC(features_test)

        # loss definition
        if args.cls_par > 0:
            #pred = mem_label[tar_idx].long()

            beta = 0.8
            py, y_prime = F.softmax(outputs_test, dim=-1).max(1)
            flag = py > beta
            classifier_loss = F.cross_entropy(outputs_test[flag], y_prime[flag])

            classifier_loss *= args.cls_par
        else:
            classifier_loss = torch.tensor(0.0)
            if args.data_env != 'local':
                classifier_loss = classifier_loss.cuda()
        if args.ent:
            '''
            outputs_test:  torch.Size([32, 2])
            softmax_out:  torch.Size([32, 2])
            entropy_loss:  torch.Size([])
            msoftmax:  torch.Size([2])
            gentropy_loss:  torch.Size([])
            '''
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss += gentropy_loss
            im_loss = entropy_loss * args.ent_par
            # SHOT-IM
            classifier_loss = im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            if args.paradigm == 'MI':
                acc_t_te, pre_t_te, rec_t_te, f1_t_te, y_pred = cal_acc(dset_loaders2["Target"], netF, netC, args=args)
                print('len(dset_loaders["Target"]): ', len(dset_loaders["Target"]))
                #acc_t_te, y_pred = cal_auc(dset_loaders["Target-Imbalanced"], netF, netC, args=args)
                # log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, iter_num, max_iter, acc_t_te)
                log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%  Pre = {:.2f}%  Rec = {:.2f}%  F1 = {:.2f}%'. \
                           format(args.task_str, int(iter_num // interval_iter), \
                           int(max_iter // interval_iter), acc_t_te, pre_t_te, rec_t_te, f1_t_te)
            args.log.record(log_str)
            print(log_str)
            netF.train()

    print('Test Acc = {:.2f}%'.format(acc_t_te))
    print('Test Pre = {:.2f}%'.format(pre_t_te))
    print('Test Rec = {:.2f}%'.format(rec_t_te))
    print('Test F1 = {:.2f}%'.format(f1_t_te))

    with open('./logs/' + str(args.method) + "_pred.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(y_pred.numpy())

    gc.collect()
    torch.cuda.empty_cache()

    return acc_t_te, pre_t_te, rec_t_te, f1_t_te


if __name__ == '__main__':

    dataset1 = 'BNCI2015001'
    dataset2 = 'BNCI2014002'

    dct = pd.DataFrame(
        columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
                 's12', 's13'])

    paradigm1, N1, chn1, class_num1, time_sample_num1, sample_rate1 = 'MI', 12, 13, 2, 2561, 512
    paradigm2, N2, chn2, class_num2, time_sample_num2, sample_rate2 = 'MI', 14, 15, 2, 2561, 512

    args = argparse.Namespace(N1=N1, data_name1=dataset1,
                              N2=N2, data_name2=dataset2,
                              class_num=2, chn=3, time_sample_num=2561, layer='wn',
                              sample_rate=512, feature_deep_dim=640,
                              lr=0.001, lr_decay1=0.1, lr_decay2=1.0,
                              ent=True, gent=True, cls_par=0, ent_par=1.0, epsilon=1e-05, interval=5,
                              smooth=0, threshold=0, distance='cosine', cov_type='oas', paradigm='MI')

    args.method = 'SHOT'
    #args.method = 'SHOT-IM'
    args.backbone = 'EEGNet'

    # whether to use EA
    args.align = True
    # learning rate
    args.lr = 0.001

    # train batch size
    args.batch_size = 32

    # training epochs
    # 0 means use pretrained models from dnn.py for SFUDA
    args.max_epoch = 100

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
            source_str = 'Except_S' + str(idt + 1)
            target_str = 'S' + str(idt + 1)
            args.task_str = source_str + '_2_' + target_str
            info_str = '\n========================== Transfer to ' + target_str + ' =========================='
            print(info_str)
            my_log.record(info_str)
            args.log = my_log

            sub_acc_all[idt], sub_pre_all[idt], sub_rec_all[idt], sub_f1_all[idt] = train_target(args)
            print("_++_+_+_+_+_+_+_++__+++_+_+_++++++++++_+++++____+_+_++_+_+_+_+_")
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
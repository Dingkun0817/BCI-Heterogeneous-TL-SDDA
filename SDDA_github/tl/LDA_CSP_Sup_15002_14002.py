import sys
# sys.path.append('E:\Pycharm_BCI\Self_Study\T-TIME')
sys.path.append('/data1/ldk/T-TIME')

import random
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils.network import backbone_net
from mne.decoding import CSP
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, auc, roc_auc_score, f1_score, precision_score, recall_score
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar, read_mi_combine_tar_diff, read_mi_combine_tar_diff_supervised_2
from utils.utils_1 import lr_scheduler_full, fix_random_seed, cal_acc_comb, data_loader
from utils.loss import MultipleKernelMaximumMeanDiscrepancy, GaussianKernel

import gc

# np.random.seed(0)
def ml_classifier(approach, output_probability, train_x, train_y, test_x, return_model=None, weight=None):
    if approach == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif approach == 'LR':
        clf = LogisticRegression(max_iter=1000)
    elif approach == 'AdaBoost':
        clf = AdaBoostClassifier()
    elif approach == 'GradientBoosting':
        clf = GradientBoostingClassifier()
    elif approach == 'xgb':
        clf = XGBClassifier()
    clf.fit(train_x, train_y)
    if output_probability:
        pred_proba = clf.predict_proba(test_x)
        return pred_proba
    else:
        pred = clf.predict(test_x)
        return pred

def train_target(args):
    X_src, y_src, X_tar, y_tar, X_tar_unlabel, y_tar_unlabel = read_mi_combine_tar_diff_supervised_2(args)
    print('X_src, y_src, X_tar, y_tar, X_tar_unlabel, y_tar_unlabel:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape, X_tar_unlabel.shape, y_tar_unlabel.shape)

    train_x = np.concatenate((X_src, X_tar), axis=0)
    train_y = np.concatenate((y_src, y_tar), axis=0)
    test_x = X_tar_unlabel
    test_y = y_tar_unlabel
    print("+_++_++_++_++__+_+-===========================")
    print('train_x:', train_x.shape)
    print('train_y:', train_y.shape)
    print("+_++_++_++_++__+_+-===========================")

    print('Training/Test split before CSP:', train_x.shape, test_x.shape)
    csp = CSP(n_components=10)
    train_x_csp = csp.fit_transform(train_x, train_y)
    test_x_csp = csp.transform(test_x)
    print('Training/Test split after CSP:', train_x_csp.shape, test_x_csp.shape)
    approach = 'LDA'
    pred = ml_classifier(approach, False, train_x_csp, train_y, test_x_csp, return_model=True)
    acc_t_te = np.round(accuracy_score(test_y, pred), 5)
    pre_t_te = np.round(accuracy_score(test_y, pred), 5)
    rec_t_te = np.round(accuracy_score(test_y, pred), 5)
    f1_t_te = np.round(accuracy_score(test_y, pred), 5)


    print('Test Acc = {:.2f}%'.format(acc_t_te))
    print('Test Pre = {:.2f}%'.format(pre_t_te))
    print('Test Rec = {:.2f}%'.format(rec_t_te))
    print('Test F1 = {:.2f}%'.format(f1_t_te))

    gc.collect()
    torch.cuda.empty_cache()

    return acc_t_te, pre_t_te, rec_t_te, f1_t_te


if __name__ == '__main__':

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

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

    args.method = 'LDA_CSP'
    args.backbone = 'LDA'

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

    total_acc = []
    total_pre = []
    total_rec = []
    total_f1 = []

    for s in [1]:
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

        sub_acc_all = np.zeros(N2)
        sub_pre_all = np.zeros(N2)
        sub_rec_all = np.zeros(N2)
        sub_f1_all = np.zeros(N2)
        for idt in range(N2):     # 留一法, idt作为test, 非idt作为train
            args.idt = idt
            source_str = str(dataset1)
            target_str = str(dataset2) + '    subject ' + str(idt)
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
    dataset = source_str + ' 2 ' + target_str
    result_dct = {'dataset': dataset, 'acc_avg': total_acc_mean, 'acc_std': total_acc_std,
                                      'pre_avg': total_pre_mean, 'pre_std': total_pre_std,
                                      'rec_avg': total_rec_mean, 'rec_std': total_rec_std,
                                      'f1_avg': total_f1_mean, 'f1_std': total_f1_std}
    for i in range(len(subject_acc_mean)):
        result_dct['s' + str(i)] = subject_acc_mean[i]

    dct = dct.append(result_dct, ignore_index=True)

    # save results to csv
    dct.to_csv('./logs/' + str(args.method) + ".csv")
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
from utils.dataloader import read_mi_combine_tar, read_mi_combine_tar_diff, read_mi_combine_tar_diff_supervised_3
from utils.utils_1 import lr_scheduler_full, fix_random_seed, cal_acc_comb, data_loader
# from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from imblearn.over_sampling import RandomOverSampler

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


def apply_pca(train_x, test_x, variance_retained):
    pca = PCA(variance_retained)
    print('before PCA:', train_x.shape, test_x.shape)
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)
    print('after PCA:', train_x.shape, test_x.shape)
    print('PCA variance retained:', np.sum(pca.explained_variance_ratio_))
    return train_x, test_x


def apply_randup(train_x, train_y):
    sampler = RandomOverSampler()
    print('before Random Upsampling:', train_x.shape, train_y.shape)
    train_x, train_y = sampler.fit_resample(train_x, train_y)
    print('after Random Upsampling:', train_x.shape, train_y.shape)
    return train_x, train_y

def train_target(args):
    X_src, y_src, X_tar, y_tar, X_tar_unlabel, y_tar_unlabel = read_mi_combine_tar_diff_supervised_3(args)
    print('X_src, y_src, X_tar, y_tar, X_tar_unlabel, y_tar_unlabel:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape, X_tar_unlabel.shape, y_tar_unlabel.shape)

    train_x = np.concatenate((X_src, X_tar), axis=0)
    train_y = np.concatenate((y_src, y_tar), axis=0)
    test_x = X_tar_unlabel
    test_y = y_tar_unlabel
    print("+_++_++_++_++__+_+-===========================")
    print('train_x:', train_x.shape)
    print('train_y:', train_y.shape)
    print("+_++_++_++_++__+_+-===========================")


    '''
    xdawn
    '''
    xdcov = XdawnCovariances()
    ts = TangentSpace()
    train_cov = xdcov.fit_transform(train_x, train_y)
    train_x_xdawn = ts.fit_transform(train_cov, train_y)

    test_cov = xdcov.transform(test_x)
    test_x_xdawn = ts.transform(test_cov)
    train_x_xdawn, train_y = apply_randup(train_x_xdawn, train_y)   # Imbalanced -> Balanced
    train_x_xdawn, test_x_xdawn = apply_pca(train_x_xdawn, test_x_xdawn, 0.95)
    print("+_++_++_++_++__+_+-===========================")
    print('train_x_xdawn:', train_x_xdawn.shape)
    print('train_y:', train_y.shape)
    print('test_x_xdawn:', test_x_xdawn.shape)
    print('test_y:', test_y.shape)
    print("+_++_++_++_++__+_+-===========================")

    approach = 'LDA'
    pred = ml_classifier(approach, True, train_x_xdawn, train_y, test_x_xdawn, return_model=True)
    auc_t_te = np.round(roc_auc_score(test_y, pred[:, 1]), 5)

    print('Test Auc = {:.2f}%'.format(auc_t_te * 100))
    gc.collect()
    torch.cuda.empty_cache()

    return auc_t_te * 100

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

    args.method = 'LDA'
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
    device = torch.device("cuda" if args.data_env == 'gpu' else "cpu")

    total_auc = []

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

        sub_auc_all = np.zeros(N2)
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
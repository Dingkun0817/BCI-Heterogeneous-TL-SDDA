# -*- coding: utf-8 -*-
# @Time    : 2023/07/13
# @Author  : Siyang Li
# @File    : utils_1.py
import os.path as osp
import os
import numpy as np
import random
import torch as tr
import torch.nn as nn
import torch.utils.data
import torch.utils.data as Data
import moabb
import mne
import learn2learn as l2l
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
from sklearn.metrics import accuracy_score, auc, roc_auc_score, f1_score, precision_score, recall_score
from scipy.linalg import fractional_matrix_power
from learn2learn.data.transforms import NWays, KShots, LoadData
from tl.utils.alg_utils import EA, EA_online
from moabb.datasets import BNCI2014001, BNCI2014002, BNCI2014008, BNCI2014009, BNCI2015003, BNCI2015004, EPFLP300, \
    BNCI2014004, BNCI2015001
from moabb.paradigms import MotorImagery, P300


def split_data(data, axis, times):
    # Splitting data into multiple sections. data: (trials, channels, time_samples)
    data_split = np.split(data, indices_or_sections=times, axis=axis)
    return data_split


def dataset_to_file(dataset_name, data_save):
    moabb.set_log_level("ERROR")
    if dataset_name == 'BNCI2014001':
        dataset = BNCI2014001()
        paradigm = MotorImagery(n_classes=4)
        # (5184, 22, 1001) (5184,) 250Hz 9subjects * 4classes * (72+72)trials for 2sessions
    elif dataset_name == 'BNCI2014002':
        dataset = BNCI2014002()
        paradigm = MotorImagery(n_classes=2)
        # (2240, 15, 2561) (2240,) 512Hz 14subjects * 2classes * (50+30)trials * 2sessions(not namely separately)
    elif dataset_name == 'BNCI2014004':
        dataset = BNCI2014004()
        paradigm = MotorImagery(n_classes=2)
        # (6520, 3, 1126) (6520,) 250Hz 9subjects * 2classes * (?)trials * 5sessions
    elif dataset_name == 'BNCI2015001':
        dataset = BNCI2015001()
        paradigm = MotorImagery(n_classes=2)
        # (5600, 13, 2561) (5600,) 512Hz 12subjects * 2 classes * (200 + 200 + (200 for Subj 8/9/10/11)) trials * (2/3)sessions
    elif dataset_name == 'MI1':
        info = None
        return info
        # (1400, 59, 300) (1400,) 100Hz 7subjects * 2classes * 200trials * 1session
    elif dataset_name == 'BNCI2015004':
        dataset = BNCI2015004()
        paradigm = MotorImagery(n_classes=2)
        # [160, 160, 160, 150 (80+70), 160, 160, 150 (80+70), 160, 160]
        # (1420, 30, 1793) (1420,) 256Hz 9subjects * 2classes * (80+80/70)trials * 2sessions
    elif dataset_name == 'BNCI2014008':
        dataset = BNCI2014008()
        paradigm = P300()
        # (33600, 8, 257) (33600,) 256Hz 8subjects 4200 trials * 1session
    elif dataset_name == 'BNCI2014009':
        dataset = BNCI2014009()
        paradigm = P300()
        # (17280, 16, 206) (17280,) 256Hz 10subjects 1728 trials * 3sessions
    elif dataset_name == 'BNCI2015003':
        dataset = BNCI2015003()
        paradigm = P300()
        # (25200, 8, 206) (25200,) 256Hz 10subjects 2520 trials * 1session
    elif dataset_name == 'EPFLP300':
        dataset = EPFLP300()
        paradigm = P300()
        # (25200, 8, 206) (25200,) 256Hz 10subjects 1session
    elif dataset_name == 'ERN':
        ch_names = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
                    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2',
                    'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3',
                    'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'POz', 'P08', 'O1', 'O2']
        info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types=['eeg'] * 56)
        return info
        # (5440, 56, 260) (5440,) 200Hz 16subjects 1session
    # SEED (152730, 62, 5*DE*)  (152730,) 200Hz 15subjects 3sessions

    if data_save:
        print('preparing data...')
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:])
        ar_unique, cnts = np.unique(labels, return_counts=True)
        print("labels:", ar_unique)
        print("Counts:", cnts)
        print(X.shape, labels.shape)
        np.save('./data/' + dataset_name + '/X', X)
        np.save('./data/' + dataset_name + '/labels', labels)
        meta.to_csv('./data/' + dataset_name + '/meta.csv')
    else:
        if isinstance(paradigm, MotorImagery):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:], return_epochs=True)
            return X.info
        elif isinstance(paradigm, P300):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:], return_epochs=True)
            return X.info


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def fix_random_seed(SEED):
    tr.manual_seed(SEED)
    tr.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def create_folder(dir_name, data_env, win_root):
    if not osp.exists(dir_name):
        os.system('mkdir -p ' + dir_name)
    if not osp.exists(dir_name):
        if data_env == 'gpu':
            os.mkdir(dir_name)
        elif data_env == 'local':
            os.makedirs(win_root + dir_name)


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def lr_scheduler_full(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def cal_acc(loader, netF, netC, args=None):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            if args.data_env != 'local':
                inputs = inputs.cuda()
            labels = data[1].float()
            outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()

    '''
    accuracy_score, f1_score, precision_score, recall_score
    '''
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred)
    recall = recall_score(true, pred)
    f1 = f1_score(true, pred)

    return accuracy * 100, precision * 100, recall * 100, f1 * 100, all_output



def cal_bca(loader, netF, netC):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cuda()
            labels = data[1].float()
            outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    bca = balanced_accuracy_score(true, pred)
    return bca * 100, all_output


def cal_auc(loader, netF, netC):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cuda()
            labels = data[1].float()
            outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    true = all_label.cpu()
    pred = all_output[:, 1].detach().numpy()
    auc = roc_auc_score(true, pred)
    return auc * 100, all_output


def cal_acc_comb(loader, model, flag=True, fc=None, args=None):
    start_test = True
    model.eval()
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            if args.data_env != 'local':
                inputs = inputs.cuda()
            inputs = inputs
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    '''
    accuracy_score, f1_score, precision_score, recall_score
    '''
    # print("====================____________________++++++++++++++")
    # torch.set_printoptions(threshold=5000)
    # print(true.sum())
    # print(pred)
    # print("====================____________________++++++++++++++")
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred)
    recall = recall_score(true, pred)
    f1 = f1_score(true, pred)

    return accuracy * 100, precision * 100, recall * 100, f1 * 100, all_output


def convert_label(labels, axis, threshold):
    # Converting labels to 0 or 1, based on a certain threshold
    label_01 = np.where(labels > threshold, 1, 0)
    # print(label_01)
    return label_01


def cal_score_online(loader, model, args):
    y_true = []
    y_pred = []
    model.eval()
    # initialize test reference matrix for Incremental EA
    if args.align:
        R = 0
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cpu()
            labels = data[1]
            if i == 0:
                data_cum = inputs.float().cpu()
            else:
                data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)

            if args.align:
                # update reference matrix
                R = EA_online(inputs.reshape(args.chn, args.time_sample_num), R, i)
                sqrtRefEA = fractional_matrix_power(R, -0.5)
                # transform current test sample
                inputs = np.dot(sqrtRefEA, inputs)
                inputs = inputs.reshape(1, 1, args.chn, args.time_sample_num)

            inputs = torch.from_numpy(inputs).to(torch.float32)
            if args.data_env != 'local':
                inputs = inputs.cuda()
            _, outputs = model(inputs)
            outputs = outputs.float().cpu()
            labels = labels.float().cpu()
            _, predict = tr.max(outputs, 1)
            pred = tr.squeeze(predict).float()
            y_pred.append(pred.item())
            y_true.append(labels.item())

            if i == 0:
                all_output = outputs.float().cpu()
                all_label = labels.float()
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)

    if args.balanced:
        acc_scores = accuracy_score(y_true, pred)
        # pre_scores = precision_score(y_true, pred)
        # rec_scores = recall_score(y_true, pred)
        # f1_scores = f1_score(y_true, pred)
    else:
        all_output = nn.Softmax(dim=1)(all_output)
        true = all_label.cpu()
        pred = all_output[:, 1].detach().numpy()
        score = roc_auc_score(true, pred)
    # imbalance的话只用AUC指标，所以返回值只会有一个
    return acc_scores * 100
    # pre_scores * 100, rec_scores * 100, f1_scores * 100


def cal_auc_comb(loader, model, flag=True, fc=None, args=None):
    start_test = True
    model.eval()
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            if args.data_env != 'local':
                inputs = inputs.cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    # _, predict = tr.max(all_output, 1)
    # pred = tr.squeeze(predict).float()
    print("-----------------------")
    print(all_output)
    true = all_label.cpu()
    pred = all_output[:, 1].detach().numpy()
    print(pred)
    print("+++++++++++++++++++++++++++")
    auc = roc_auc_score(true, pred)

    return auc * 100, all_output


def cal_metrics_multisource(loader, nets, args, metrics):
    # mode 'avg', 'vote'
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            all_probs = None
            for i in range(args.N - 1):
                if args.data_env != 'local':
                    x = x.cuda()
                    y = y.cuda()
                outputs = nets[i][0](x)
                _, outputs = nets[i][1](outputs)
                predicted_probs = torch.nn.functional.softmax(outputs, dim=1)
                if all_probs is None:
                    all_probs = torch.zeros((x.shape[0], args.class_num))
                    if args.data_env != 'local':
                        all_probs = all_probs.cuda()
                else:
                    all_probs += predicted_probs.reshape(x.shape[0], args.class_num)

                _, predicted = torch.max(predicted_probs, 1)

                if args.mode == 'vote':
                    votes = torch.zeros((x.shape[0], args.class_num))
                    if args.data_env != 'local':
                        votes = votes.cuda()
                    for i in range(x.shape[0]):
                        votes[i, predicted[i]] += 1
            if args.mode == 'vote':
                _, predicted = torch.max(votes, 1)  # VOTING
            if args.mode == 'avg':
                _, predicted = torch.max(all_probs, 1)  # AVERAGING
            y_true.append(y.cpu())
            y_pred.append(predicted.cpu())
    score = metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(-1, )[0]
    return score * 100


def data_alignment(X, num_subjects, dataset):
    '''
    :param X: np array, EEG data
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    out = []
    if dataset == "BNCI2014004" or dataset == 'BNCI2014004_filter':
        print('before EA:', X.shape)
        for i in range(num_subjects):
            if(i == 0):
                tmp_x = EA(X[0:720, :, :])
            if(i == 1):
                tmp_x = EA(X[720:1400, :, :])
            if(i == 2):
                tmp_x = EA(X[1400:2120, :, :])
            if(i == 3):
                tmp_x = EA(X[2120:2860, :, :])
            if(i == 4):
                tmp_x = EA(X[2860:3600, :, :])
            if(i == 5):
                tmp_x = EA(X[3600:4320, :, :])
            if(i == 6):
                tmp_x = EA(X[4320:5040, :, :])
            if(i == 7):
                tmp_x = EA(X[5040:5800, :, :])
            if(i == 8):
                tmp_x = EA(X[5800:6520, :, :])
            out.append(tmp_x)
        X = np.concatenate(out, axis=0)
        print('after EA:', X.shape)

    else:
        print('before EA:', X.shape)
        out = []
        for i in range(num_subjects):
            tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
            out.append(tmp_x)
        X = np.concatenate(out, axis=0)
        print('after EA:', X.shape)
    return X

def data_alignment_session(X, num_subjects, dataset):
    '''
    :param X: np array, EEG data
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    out = []
    if dataset == "BNCI2014004" or dataset == 'BNCI2014004_filter':
        print('before sessionEA:', X.shape)
        for i in range(num_subjects):
            if(i == 0):
                part_1 = EA(X[0:120, :, :])
                part_2 = EA(X[120:240, :, :])
                part_3 = EA(X[240:400, :, :])
                part_4 = EA(X[400:560, :, :])
                part_5 = EA(X[560:720, :, :])
                tmp_x = np.concatenate([part_1, part_2, part_3, part_4, part_5], axis=0)
                print(tmp_x.shape)
            if(i == 1):
                part_1 = EA(X[720:840, :, :])
                part_2 = EA(X[840:960, :, :])
                part_3 = EA(X[960:1120, :, :])
                part_4 = EA(X[1120:1240, :, :])
                part_5 = EA(X[1240:1400, :, :])
                tmp_x = np.concatenate([part_1, part_2, part_3, part_4, part_5], axis=0)
                print(tmp_x.shape)
            if(i == 2):
                part_1 = EA(X[1400:1520, :, :])
                part_2 = EA(X[1520:1640, :, :])
                part_3 = EA(X[1640:1800, :, :])
                part_4 = EA(X[1800:1960, :, :])
                part_5 = EA(X[1960:2120, :, :])
                tmp_x = np.concatenate([part_1, part_2, part_3, part_4, part_5], axis=0)
                print(tmp_x.shape)
            if(i == 3):
                part_1 = EA(X[2120:2240, :, :])
                part_2 = EA(X[2240:2380, :, :])
                part_3 = EA(X[2380:2540, :, :])
                part_4 = EA(X[2540:2700, :, :])
                part_5 = EA(X[2700:2860, :, :])
                tmp_x = np.concatenate([part_1, part_2, part_3, part_4, part_5], axis=0)
                print(tmp_x.shape)
            if(i == 4):
                part_1 = EA(X[2860:2980, :, :])
                part_2 = EA(X[2980:3120, :, :])
                part_3 = EA(X[3120:3280, :, :])
                part_4 = EA(X[3280:3440, :, :])
                part_5 = EA(X[3440:3600, :, :])
                tmp_x = np.concatenate([part_1, part_2, part_3, part_4, part_5], axis=0)
                print(tmp_x.shape)
            if(i == 5):
                part_1 = EA(X[3600:3720, :, :])
                part_2 = EA(X[3720:3840, :, :])
                part_3 = EA(X[3840:4000, :, :])
                part_4 = EA(X[4000:4160, :, :])
                part_5 = EA(X[4160:4320, :, :])
                tmp_x = np.concatenate([part_1, part_2, part_3, part_4, part_5], axis=0)
                print(tmp_x.shape)
            if(i == 6):
                part_1 = EA(X[4320:4440, :, :])
                part_2 = EA(X[4440:4560, :, :])
                part_3 = EA(X[4560:4720, :, :])
                part_4 = EA(X[4720:4880, :, :])
                part_5 = EA(X[4880:5040, :, :])
                tmp_x = np.concatenate([part_1, part_2, part_3, part_4, part_5], axis=0)
                print(tmp_x.shape)
            if(i == 7):
                part_1 = EA(X[5040:5200, :, :])
                part_2 = EA(X[5200:5320, :, :])
                part_3 = EA(X[5320:5480, :, :])
                part_4 = EA(X[5480:5640, :, :])
                part_5 = EA(X[5640:5800, :, :])
                tmp_x = np.concatenate([part_1, part_2, part_3, part_4, part_5], axis=0)
                print(tmp_x.shape)
            if(i == 8):
                part_1 = EA(X[5800:5920, :, :])
                part_2 = EA(X[5920:6040, :, :])
                part_3 = EA(X[6040:6200, :, :])
                part_4 = EA(X[6200:6360, :, :])
                part_5 = EA(X[6360:6520, :, :])
                tmp_x = np.concatenate([part_1, part_2, part_3, part_4, part_5], axis=0)
                print(tmp_x.shape)
            out.append(tmp_x)
        X = np.concatenate(out, axis=0)
        print('after sessionEA:', X.shape)
    elif dataset == "BNCI2014002":
        print('BNCI2014002 before EA:', X.shape)
        out = []
        num_samples = X.shape[0]
        ss = 20

        for i in range(0, num_samples, ss):
            tmp_x = EA(X[i:i + ss, :, :])
            out.append(tmp_x)

        X = np.concatenate(out, axis=0)
        print('BNCI2014002 after EA:', X.shape)
    elif dataset == "BNCI2014009":
        print('BNCI2014009 before EA:', X.shape)
        out = []
        num_samples = X.shape[0]
        ss = 576

        for i in range(0, num_samples, ss):
            tmp_x = EA(X[i:i + ss, :, :])
            out.append(tmp_x)

        X = np.concatenate(out, axis=0)
        print('BNCI2014009 after EA:', X.shape)


    else:
        print('before EA:', X.shape)
        out = []
        for i in range(num_subjects):
            tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
            out.append(tmp_x)
        X = np.concatenate(out, axis=0)
        print('after EA:', X.shape)
    return X


def data_loader(Xs=None, Ys=None, Xt=None, Yt=None, args=None):
    # cross-subject loader
    dset_loaders = {}
    train_bs = args.batch_size

    Xt_copy = Xt
    '''
    EA对齐在读数据  read_mi_combine_tar_diff  的时候就对齐了
    '''
    # if args.align:
    #     Xs = data_alignment(Xs, args.N1 - 1, args.data1, True)   # N1是源域被试数, N2是目标域被试数
    #     Xt = data_alignment(Xt, args.idt, args.data2, False)

    Xs, Ys = tr.from_numpy(Xs).to(tr.float32), tr.from_numpy(Ys.reshape(-1, )).to(tr.long)
    Xs = Xs.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xs = Xs.permute(0, 3, 1, 2)

    Xt, Yt = tr.from_numpy(Xt).to(tr.float32), tr.from_numpy(Yt.reshape(-1, )).to(tr.long)
    Xt = Xt.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt = Xt.permute(0, 3, 1, 2)

    Xt_copy = tr.from_numpy(Xt_copy).to(tr.float32)
    Xt_copy = Xt_copy.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt_copy = Xt_copy.permute(0, 3, 1, 2)

    if args.data_env != 'local':
        Xs, Ys, Xt, Yt, Xt_copy = Xs.cuda(), Ys.cuda(), Xt.cuda(), Yt.cuda(), Xt_copy.cuda()
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("args.data_env: ", args.data_env)
    print("Xs size:", Xs.size())
    print("Ys size:", Ys.size())
    print("Xt size:", Xt.size())
    print("Yt size:", Yt.size())
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    data_src = Data.TensorDataset(Xs, Ys)
    data_tar = Data.TensorDataset(Xt, Yt)


    data_tar_online = Data.TensorDataset(Xt_copy, Yt)

    # for TL train
    dset_loaders["source"] = Data.DataLoader(data_src, batch_size=train_bs, shuffle=False, drop_last=True)
    dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=False, drop_last=True)
    # dset_loaders["source"] = Data.DataLoader(data_src, batch_size=train_bs, shuffle=True, drop_last=True)
    # dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=True, drop_last=True)

    # for TL test
    dset_loaders["Source"] = Data.DataLoader(data_src, batch_size=train_bs * 3, shuffle=False, drop_last=False)
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    # for online TL test
    dset_loaders["Target-Online"] = Data.DataLoader(data_tar_online, batch_size=1, shuffle=False, drop_last=False)
    # for online imbalanced dataset
    # only implemented for binary (class_num=2) for now

    class_0_ids = torch.where(Yt == 0)[0][:len(Yt) // 2]
    class_1_ids = torch.where(Yt == 1)[0][:len(Yt) // 4]
    all_ids = torch.cat([class_0_ids, class_1_ids])
    if args.data_env != 'local':
        data_tar_imb = Data.TensorDataset(Xt_copy[all_ids].cuda(), Yt[all_ids].cuda())
    else:
        data_tar_imb = Data.TensorDataset(Xt_copy[all_ids], Yt[all_ids])
    dset_loaders["Target-Online-Imbalanced"] = Data.DataLoader(data_tar_imb, batch_size=1, shuffle=True,
                                                               drop_last=False)
    dset_loaders["target-Imbalanced"] = Data.DataLoader(data_tar_imb, batch_size=train_bs, shuffle=True, drop_last=True)
    dset_loaders["Target-Imbalanced"] = Data.DataLoader(data_tar_imb, batch_size=train_bs * 3, shuffle=True,
                                                        drop_last=False)

    return dset_loaders


def data_loader_multisource(Xs=None, Ys=None, Xt=None, Yt=None, args=None):
    # multi-source cross-subject loader
    dset_loaders = {}
    train_bs = args.batch_size

    Xt_copy = Xt
    if args.align:
        for i in range(len(Xs)):
            Xs[i] = data_alignment(Xs[i], 1, args)
        Xt = data_alignment(Xt, 1, args)

    for i in range(len(Xs)):
        Xs[i], Ys[i] = tr.from_numpy(Xs[i]).to(
            tr.float32), tr.from_numpy(Ys[i].reshape(-1, )).to(tr.long)
        Xs[i] = Xs[i].unsqueeze_(3)
        if 'EEGNet' in args.backbone:
            Xs[i] = Xs[i].permute(0, 3, 1, 2)

    Xt, Yt = tr.from_numpy(Xt).to(
        tr.float32), tr.from_numpy(Yt.reshape(-1, )).to(tr.long)
    Xt = Xt.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt = Xt.permute(0, 3, 1, 2)

    Xt_copy = tr.from_numpy(Xt_copy).to(
        tr.float32)
    Xt_copy = Xt_copy.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt_copy = Xt_copy.permute(0, 3, 1, 2)

    sources_ms = []
    for i in range(args.N - 1):
        if args.data_env != 'local':
            Xs[i], Ys[i] = Xs[i].cuda(), Ys[i].cuda()
        source = Data.TensorDataset(Xs[i], Ys[i])
        sources_ms.append(source)

    if args.data_env != 'local':
        Xt, Yt, Xt_copy = Xt.cuda(), Yt.cuda(), Xt_copy.cuda()

    data_tar = Data.TensorDataset(Xt, Yt)

    data_tar_online = Data.TensorDataset(Xt_copy, Yt)

    # for TL test
    dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=True, drop_last=True)
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    # for online TL test
    dset_loaders["Target-Online"] = Data.DataLoader(data_tar_online, batch_size=1, shuffle=False, drop_last=False)

    # for multi-sources
    loader_arr = []
    for i in range(args.N - 1):
        loader = Data.DataLoader(sources_ms[i], batch_size=train_bs, shuffle=True, drop_last=True)
        loader_arr.append(loader)
    dset_loaders["sources"] = loader_arr

    loader_arr_S = []
    for i in range(args.N - 1):
        loader = Data.DataLoader(sources_ms[i], batch_size=train_bs, shuffle=False, drop_last=False)
        loader_arr_S.append(loader)
    dset_loaders["Sources"] = loader_arr_S

    return dset_loaders


def data_loader_without_tar(Xs=None, Ys=None, args=None):
    # no target process
    dset_loaders = {}
    train_bs = args.batch_size

    if args.align:
        Xs = data_alignment(Xs, args.N - 1, args)

    Xs, Ys = tr.from_numpy(Xs).to(
        tr.float32), tr.from_numpy(Ys.reshape(-1, )).to(tr.long)
    Xs = Xs.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xs = Xs.permute(0, 3, 1, 2)

    if args.data_env != 'local':
        Xs, Ys = Xs.cuda(), Ys.cuda()

    data_src = Data.TensorDataset(Xs, Ys)

    # for TL train
    dset_loaders["source"] = Data.DataLoader(data_src, batch_size=train_bs, shuffle=True, drop_last=True)

    # for TL test
    dset_loaders["Source"] = Data.DataLoader(data_src, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    return dset_loaders


def data_loader_split(Xs=None, Ys=None, Xt=None, Yt=None, args=None):
    # within-subject loader
    dset_loaders = {}
    train_bs = args.batch_size

    if args.align:
        Xs = data_alignment(Xs, args.N, args)
        Xt = data_alignment(Xt, args.N, args)

    Xs, Ys = tr.from_numpy(Xs).to(
        tr.float32), tr.from_numpy(Ys.reshape(-1, )).to(tr.long)
    Xs = Xs.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xs = Xs.permute(0, 3, 1, 2)

    Xt, Yt = tr.from_numpy(Xt).to(
        tr.float32), tr.from_numpy(Yt.reshape(-1, )).to(tr.long)
    Xt = Xt.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt = Xt.permute(0, 3, 1, 2)

    if args.data_env != 'local':
        Xs, Ys, Xt, Yt = Xs.cuda(), Ys.cuda(), Xt.cuda(), Yt.cuda()

    data_src = Data.TensorDataset(Xs, Ys)
    data_tar = Data.TensorDataset(Xt, Yt)

    # for TL train
    dset_loaders["source"] = Data.DataLoader(data_src, batch_size=train_bs, shuffle=True, drop_last=True)

    # for TL test
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    return dset_loaders
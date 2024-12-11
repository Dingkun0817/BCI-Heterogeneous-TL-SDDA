# -*- coding: utf-8 -*-
# @Time    : 2023/7/11
# @Author  : Siyang Li
# @File    : dataloader.py
import numpy as np
from sklearn import preprocessing
from tl.utils.data_utils import traintest_split_cross_subject, traintest_split_domain_classifier, traintest_split_multisource, traintest_split_domain_classifier_pretest, traintest_split_multisource
from tl.utils.utils_1 import data_alignment, data_alignment_session
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

def data_process(dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    if dataset == 'BNCI2014001-4':
        X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
    else:
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001' or dataset == 'BNCI2014001_filter' or dataset == 'BNCI2014001_short':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    if dataset == 'BNCI2014001_23':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes ['right_hand', 'feet']
        indices = []
        for i in range(len(y)):
            if y[i] in ['right_hand', 'feet']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session train, remove session test
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(100) + (160 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014004' or dataset == 'BNCI2014004_filter' or dataset == 'BNCI2014004_short':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 3
    elif dataset == 'BNCI2014008':
        paradigm = 'MI'
        num_subjects = 8
        sample_rate = 256
        ch_num = 8
    elif dataset == 'BNCI2014009':
        paradigm = 'MI'
        num_subjects = 10
        sample_rate = 256
        ch_num = 16

    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))

        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def data_process_secondsession(dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    if dataset == 'BNCI2014001-4':
        X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
    else:
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i) + 288) # use second sessions
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session train, remove session test
        indices = []
        for i in range(num_subjects):
            #indices.append(np.arange(100) + (160 * i))
            indices.append(np.arange(60) + (160 * i) + 100) # use second sessions
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            # use second sessions
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))

        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def read_mi_combine_tar(args):
    if 'continual' in args.method:  # TODO
        # Continual TTA
        X, y, num_subjects, paradigm, sample_rate, ch_num = data_process_secondsession(args.data)
    else:
        X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)

    src_data, src_label, tar_data, tar_label = traintest_split_cross_subject(args.data, X, y, num_subjects, args.idt)

    return src_data, src_label, tar_data, tar_label

def read_mi_combine_tar_diff(args):
    X_1, y_1, num_subjects_1, paradigm_1, sample_rate_1, ch_num_1 = data_process(args.data1)
    X_2, y_2, num_subjects_2, paradigm_2, sample_rate_2, ch_num_2 = data_process(args.data2)

    if args.align:
        print("-------   START EA: -------")
        X_1 = data_alignment(X_1, args.N1, args.data1)
        # X_2 = data_alignment(X_2, args.N2, args.data2)
        X_2 = data_alignment(X_2, args.N2, args.data2)
        print("-------   FINISH EA: -------")

    X_1 = X_1[:, [7, 9, 11], :]
    X_2 = X_2[:, :, 0:1001]

    train_x = X_1
    train_y = y_1

    print('======================subject {}============================='.format(args.idt))

    '''  
    BNCI2014004
    '''
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

    # src_data, src_label, tar_data, tar_label = traintest_split_cross_subject(args.data, X, y, num_subjects, args.idt)
    src_data = train_x
    src_label = train_y
    tar_data = test_x
    tar_label = test_y
    return src_data, src_label, tar_data, tar_label


def read_mi_combine_tar_diff_supervised(args):
    X_1, y_1, num_subjects_1, paradigm_1, sample_rate_1, ch_num_1 = data_process(args.data1)
    X_2, y_2, num_subjects_2, paradigm_2, sample_rate_2, ch_num_2 = data_process(args.data2)

    if args.align:
        print("-------   START EA: -------")
        X_1 = data_alignment(X_1, args.N1, args.data1)
        X_2 = data_alignment(X_2, args.N2, args.data2)
        # X_2 = data_alignment_session(X_2, args.N2, args.data2)
        print("-------   FINISH EA: -------")

    X_1 = X_1[:, [7, 9, 11], :]
    X_2 = X_2[:, :, 0:1001]

    train_x = X_1
    train_y = y_1

    print('======================subject {}============================='.format(args.idt))

    '''  
    BNCI2014004
    '''
    if args.idt == 0:
        test_x = X_2[0:32, :, :]
        test_y = y_2[0:32]
        test_xx = X_2[32:720, :, :]
        test_yy = y_2[32:720]
    if args.idt == 1:
        test_x = X_2[720:752, :, :]
        test_y = y_2[720:752]
        test_xx = X_2[752:1400, :, :]
        test_yy = y_2[752:1400]
    if args.idt == 2:
        test_x = X_2[1400:1432, :, :]
        test_y = y_2[1400:1432]
        test_xx = X_2[1432:2120, :, :]
        test_yy = y_2[1432:2120]
    if args.idt == 3:
        test_x = X_2[2120:2152, :, :]
        test_y = y_2[2120:2152]
        test_xx = X_2[2152:2860, :, :]
        test_yy = y_2[2152:2860]
    if args.idt == 4:
        test_x = X_2[2860:2892, :, :]
        test_y = y_2[2860:2892]
        test_xx = X_2[2892:3600, :, :]
        test_yy = y_2[2892:3600]
    if args.idt == 5:
        test_x = X_2[3600:3632, :, :]
        test_y = y_2[3600:3632]
        test_xx = X_2[3632:4320, :, :]
        test_yy = y_2[3632:4320]
    if args.idt == 6:
        test_x = X_2[4320:4352, :, :]
        test_y = y_2[4320:4352]
        test_xx = X_2[4352:5040, :, :]
        test_yy = y_2[4352:5040]
    if args.idt == 7:
        test_x = X_2[5040:5072, :, :]
        test_y = y_2[5040:5072]
        test_xx = X_2[5072:5800, :, :]
        test_yy = y_2[5072:5800]
    if args.idt == 8:
        test_x = X_2[5800:5832, :, :]
        test_y = y_2[5800:5832]
        test_xx = X_2[5832:6520, :, :]
        test_yy = y_2[5832:6520]
    print('train_x, train_y, test_x, test_y, test_xx, test_yy', train_x.shape, train_y.shape, test_x.shape, test_y.shape, test_xx.shape, test_yy.shape)

    # src_data, src_label, tar_data, tar_label = traintest_split_cross_subject(args.data, X, y, num_subjects, args.idt)
    src_data = train_x
    src_label = train_y
    tar_data = test_x
    tar_label = test_y
    tar_data_unlabel = test_xx
    tar_label_unlabel = test_yy
    return src_data, src_label, tar_data, tar_label, tar_data_unlabel, tar_label_unlabel

def read_mi_combine_tar_diff_supervised_2(args):
    X_1, y_1, num_subjects_1, paradigm_1, sample_rate_1, ch_num_1 = data_process(args.data1)
    X_2, y_2, num_subjects_2, paradigm_2, sample_rate_2, ch_num_2 = data_process(args.data2)
    #
    if args.align:
        print("-------   START EA: -------")
        X_1 = data_alignment(X_1, args.N1, args.data1)
        X_2 = data_alignment(X_2, args.N2, args.data2)
        # X_2 = data_alignment_session(X_2, args.N2, args.data2)
        print("-------   FINISH EA: -------")

    X_1 = X_1[:, [4, 6, 8], :]
    X_2 = X_2[:, [4, 7, 10], :]

    train_x = X_1
    train_y = y_1

    print('======================subject {}============================='.format(args.idt))

    '''  
    BNCI2014002
    '''
    if args.idt == 0:
        test_x = X_2[0:32, :, :]
        test_y = y_2[0:32]
        test_xx = X_2[32:100, :, :]
        test_yy = y_2[32:100]
    if args.idt == 1:
        test_x = X_2[100:132, :, :]
        test_y = y_2[100:132]
        test_xx = X_2[132:200, :, :]
        test_yy = y_2[132:200]
    if args.idt == 2:
        test_x = X_2[200:232, :, :]
        test_y = y_2[200:232]
        test_xx = X_2[232:300, :, :]
        test_yy = y_2[232:300]
    if args.idt == 3:
        test_x = X_2[300:332, :, :]
        test_y = y_2[300:332]
        test_xx = X_2[332:400, :, :]
        test_yy = y_2[332:400]
    if args.idt == 4:
        test_x = X_2[400:432, :, :]
        test_y = y_2[400:432]
        test_xx = X_2[432:500, :, :]
        test_yy = y_2[432:500]
    if args.idt == 5:
        test_x = X_2[500:532, :, :]
        test_y = y_2[500:532]
        test_xx = X_2[532:600, :, :]
        test_yy = y_2[532:600]
    if args.idt == 6:
        test_x = X_2[600:632, :, :]
        test_y = y_2[600:632]
        test_xx = X_2[632:700, :, :]
        test_yy = y_2[632:700]
    if args.idt == 7:
        test_x = X_2[700:732, :, :]
        test_y = y_2[700:732]
        test_xx = X_2[732:800, :, :]
        test_yy = y_2[732:800]
    if args.idt == 8:
        test_x = X_2[800:832, :, :]
        test_y = y_2[800:832]
        test_xx = X_2[832:900, :, :]
        test_yy = y_2[832:900]
    if args.idt == 9:
        test_x = X_2[900:932, :, :]
        test_y = y_2[900:932]
        test_xx = X_2[932:1000, :, :]
        test_yy = y_2[932:1000]
    if args.idt == 10:
        test_x = X_2[1000:1032, :, :]
        test_y = y_2[1000:1032]
        test_xx = X_2[1032:1100, :, :]
        test_yy = y_2[1032:1100]
    if args.idt == 11:
        test_x = X_2[1100:1132, :, :]
        test_y = y_2[1100:1132]
        test_xx = X_2[1132:1200, :, :]
        test_yy = y_2[1132:1200]
    if args.idt == 12:
        test_x = X_2[1200:1232, :, :]
        test_y = y_2[1200:1232]
        test_xx = X_2[1232:1300, :, :]
        test_yy = y_2[1232:1300]
    if args.idt == 13:
        test_x = X_2[1300:1332, :, :]
        test_y = y_2[1300:1332]
        test_xx = X_2[1332:1400, :, :]
        test_yy = y_2[1332:1400]
    print('train_x, train_y, test_x, test_y, test_xx, test_yy', train_x.shape, train_y.shape, test_x.shape, test_y.shape, test_xx.shape, test_yy.shape)

    # src_data, src_label, tar_data, tar_label = traintest_split_cross_subject(args.data, X, y, num_subjects, args.idt)
    src_data = train_x
    src_label = train_y
    tar_data = test_x
    tar_label = test_y
    tar_data_unlabel = test_xx
    tar_label_unlabel = test_yy
    return src_data, src_label, tar_data, tar_label, tar_data_unlabel, tar_label_unlabel


def read_mi_combine_tar_diff_supervised_3(args):
    X_1, y_1, num_subjects_1, paradigm_1, sample_rate_1, ch_num_1 = data_process(args.data1)
    X_2, y_2, num_subjects_2, paradigm_2, sample_rate_2, ch_num_2 = data_process(args.data2)

    if args.align:
        print("-------   START EA: -------")
        X_1 = data_alignment(X_1, args.N1, args.data1)
        X_2 = data_alignment(X_2, args.N2, args.data2)
        # X_2 = data_alignment_session(X_2, args.N2, args.data2)
        print("-------   FINISH EA: -------")

    X_1 = X_1[:, 0:8, :]
    X_2 = X_2[:, :, 0:206]

    pos_indices = np.where(y_1 == 1)[0]
    neg_indices = np.where(y_1 == 0)[0]
    selected_pos_indices = pos_indices
    # selected_neg_indices = neg_indices[:2880]
    selected_neg_indices = neg_indices
    selected_indices = np.concatenate((selected_pos_indices, selected_neg_indices))
    np.random.seed(42)
    np.random.shuffle(selected_indices)  # 打乱索引
    X_1 = X_1[selected_indices]
    y_1 = y_1[selected_indices]

    train_x = X_1
    train_y = y_1

    print('======================subject {}============================='.format(args.idt))

    '''  
    BNCI2014008
    '''
    data_subjects = np.split(X_2, indices_or_sections=8, axis=0)
    labels_subjects = np.split(y_2, indices_or_sections=8, axis=0)   #  把每个被试分出来
    X_2 = data_subjects.pop(args.idt)
    y_2 = labels_subjects.pop(args.idt)

    pos_indices = np.where(y_2 == 1)[0]
    neg_indices = np.where(y_2 == 0)[0]
    selected_pos_indices = pos_indices
    # selected_neg_indices = neg_indices[:700]
    selected_neg_indices = neg_indices
    selected_indices = np.concatenate((selected_pos_indices, selected_neg_indices))
    np.random.shuffle(selected_indices)
    X_2 = X_2[selected_indices]
    y_2 = y_2[selected_indices]


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



    print('train_x, train_y, test_x, test_y, test_xx, test_yy', train_x.shape, train_y.shape, test_x.shape, test_y.shape, test_xx.shape, test_yy.shape)

    # src_data, src_label, tar_data, tar_label = traintest_split_cross_subject(args.data, X, y, num_subjects, args.idt)
    src_data = train_x
    src_label = train_y
    tar_data = test_x
    tar_label = test_y
    tar_data_unlabel = test_xx
    tar_label_unlabel = test_yy
    return src_data, src_label, tar_data, tar_label, tar_data_unlabel, tar_label_unlabel

'''
  5个点取平均
'''
def read_mi_combine_tar_ldk_avg(args):
    X_1, y_1, num_subjects_1, paradigm_1, sample_rate_1, ch_num_1 = data_process(args.data1)
    X_2, y_2, num_subjects_2, paradigm_2, sample_rate_2, ch_num_2 = data_process(args.data2)

    if args.align:
        print("-------   START EA: -------")
        X_1 = data_alignment(X_1, args.N1, args.data1)
        X_2 = data_alignment(X_2, args.N2, args.data2)
        print("-------   FINISH EA: -------")

    # X_1 = X_1[:, [7, 9, 11], :]

    channel_indices = [[1, 6, 7, 8, 13],
                       [4, 8, 9, 10, 15],
                       [5, 10, 11, 12, 17]]
    new_X_1 = np.zeros((X_1.shape[0], 3, X_1.shape[2]), dtype=X_1.dtype)
    for i, indices in enumerate(channel_indices):
        channel_sum = np.zeros((X_1.shape[0], 1, X_1.shape[2]), dtype=X_1.dtype)
        for idx in indices:
            channel_sum += X_1[:, idx, :][:, np.newaxis, :]
        new_X_1[:, i, :] = channel_sum.squeeze(1) / len(indices)
    X_1 = new_X_1
    X_2 = X_2[:, :, 0:1001]

    train_x = X_1
    train_y = y_1

    print('======================subject {}============================='.format(args.idt))
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

    # src_data, src_label, tar_data, tar_label = traintest_split_cross_subject(args.data, X, y, num_subjects, args.idt)
    src_data = train_x
    src_label = train_y
    tar_data = test_x
    tar_label = test_y
    return src_data, src_label, tar_data, tar_label
'''
5个点利用pearsonr相关系数加权平均
'''
def read_mi_combine_tar_ldk_pearson_avg(args):
    X_1, y_1, num_subjects_1, paradigm_1, sample_rate_1, ch_num_1 = data_process(args.data1)
    X_2, y_2, num_subjects_2, paradigm_2, sample_rate_2, ch_num_2 = data_process(args.data2)

    # if args.align:
    #     print("-------   START EA: -------")
    #     X_1 = data_alignment(X_1, args.N1, args.data1)
    #     X_2 = data_alignment(X_2, args.N2, args.data2)
    #     print("-------   FINISH EA: -------")

    channel_indices = [[1, 6, 7, 8, 13],
                       [4, 8, 9, 10, 15],
                       [5, 10, 11, 12, 17]]

    new_X_1 = np.zeros((X_1.shape[0], 3, X_1.shape[2]), dtype=X_1.dtype)

    for i in range(X_1.shape[0]):
        for jj in range(3):
            channel_avg = np.zeros((1, 1, X_1.shape[2]), dtype=X_1.dtype)
            if (jj == 0):
                pearsonr_values = np.array([pearsonr(X_1[i, j, :], X_1[i, 7, :])[0] for j in channel_indices[jj]])
                sum_w = 0
                for w in range(len(pearsonr_values)):
                    sum_w += pearsonr_values[w]  # 权重和
                    channel_avg = channel_avg + X_1[i, channel_indices[jj][w], :] * pearsonr_values[w]
                channel_avg = channel_avg / sum_w
            if (jj == 1):
                pearsonr_values = np.array([pearsonr(X_1[i, j, :], X_1[i, 9, :])[0] for j in channel_indices[jj]])
                sum_w = 0
                for w in range(len(pearsonr_values)):
                    sum_w += pearsonr_values[w]  # 权重和
                    channel_avg = channel_avg + X_1[i, channel_indices[jj][w], :] * pearsonr_values[w]
                channel_avg = channel_avg / sum_w
            if (jj == 2):
                pearsonr_values = np.array([pearsonr(X_1[i, j, :], X_1[i, 11, :])[0] for j in channel_indices[jj]])
                sum_w = 0
                for w in range(len(pearsonr_values)):
                    sum_w += pearsonr_values[w]  # 权重和
                    channel_avg = channel_avg + X_1[i, channel_indices[jj][w], :] * pearsonr_values[w]
                channel_avg = channel_avg / sum_w

            new_X_1[i, jj, :] = channel_avg

    X_1 = new_X_1

    if args.align:  # 之前是读完数据直接最先EA
        print("-------   START EA: -------")
        X_1 = data_alignment(X_1, args.N1, args.data1)
        X_2 = data_alignment(X_2, args.N2, args.data2)
        print("-------   FINISH EA: -------")

    X_2 = X_2[:, :, 0:1001]

    train_x = X_1
    train_y = y_1

    print('======================subject {}============================='.format(args.idt))
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

    # src_data, src_label, tar_data, tar_label = traintest_split_cross_subject(args.data, X, y, num_subjects, args.idt)
    src_data = train_x
    src_label = train_y
    tar_data = test_x
    tar_label = test_y
    return src_data, src_label, tar_data, tar_label


def read_mi_combine_domain(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)

    src_data, src_label, tar_data, tar_label = traintest_split_domain_classifier(args.data, X, y, num_subjects, args.idt)

    return src_data, src_label, tar_data, tar_label


def read_mi_combine_domain_split(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)

    src_data, src_label, tar_data, tar_label = traintest_split_domain_classifier_pretest(args.data, X, y, num_subjects, args.ratio)

    return src_data, src_label, tar_data, tar_label


def read_mi_multi_source(args):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)

    src_data, src_label, tar_data, tar_label = traintest_split_multisource(args.data, X, y, num_subjects, args.idt)

    return src_data, src_label, tar_data, tar_label


def data_normalize(fea_de, norm_type):
    if norm_type == 'zscore':
        zscore = preprocessing.StandardScaler()
        fea_de = zscore.fit_transform(fea_de)

    return fea_de

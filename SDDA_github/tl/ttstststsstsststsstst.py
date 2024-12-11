'''
22加权平均到3个点
'''

import numpy as np
from sklearn import preprocessing
from tl.utils.data_utils import traintest_split_cross_subject, traintest_split_domain_classifier, traintest_split_multisource, traintest_split_domain_classifier_pretest, traintest_split_multisource
from tl.utils.utils_1 import data_alignment
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

def data_process(dataset):
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

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num

dataset = 'BNCI2014001'
X_1, y_1, num_subjects_1, paradigm_1, sample_rate_1, ch_num_1 = data_process(dataset)


print("-------   START EA: -------")
X_1 = data_alignment(X_1, 9, 'BNCI2014001')
print("-------   FINISH EA: -------")

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
                print(pearsonr_values[w])
                channel_avg = channel_avg + X_1[i, channel_indices[jj][w], :] * pearsonr_values[w]
            print("===============================")
            channel_avg = channel_avg / sum_w

        new_X_1[i, jj, :] = channel_avg
print(new_X_1.shape)

print(pearsonr(X_1[0, 7, :], new_X_1[0, 0, :])[0])
print(pearsonr(X_1[0, 9, :], new_X_1[0, 1, :])[0])
print(pearsonr(X_1[0, 11, :], new_X_1[0, 2, :])[0])



# corr, _ = pearsonr(X_1[0, 1, :], X_1[0, 7, :])
# print(corr)
# corr, _ = pearsonr(X_1[0, 6, :], X_1[0, 7, :])
# print(corr)
# corr, _ = pearsonr(X_1[0, 8, :], X_1[0, 7, :])
# print(corr)
# corr, _ = pearsonr(X_1[0, 13, :], X_1[0, 7, :])
# print(corr)
# corr, _ = pearsonr(X_1[0, 0, :], X_1[0, 7, :])
# print(corr)
#
# print("=============")
# print(cosine_similarity(X_1[:, 1, :], X_1[:, 7, :]).shape)



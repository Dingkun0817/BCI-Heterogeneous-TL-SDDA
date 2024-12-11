
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

    if dataset == 'BNCI2014001' or dataset == 'BNCI2014001_filter':
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
    if dataset == 'BNCI2014004' or dataset == 'BNCI2014004_filter':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 3

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num

dataset1 = 'BNCI2014001_filter'
X_1, y_1, num_subjects_1, paradigm_1, sample_rate_1, ch_num_1 = data_process(dataset1)
dataset2 = 'BNCI2014004_filter'
X_2, y_2, num_subjects_2, paradigm_2, sample_rate_2, ch_num_2 = data_process(dataset2)
data1 = dataset1
data2 = dataset2
print("-------   START EA: -------")
X_1 = data_alignment(X_1, 9, data1)
X_2 = data_alignment(X_2, 9, data2)
print("-------   FINISH EA: -------========")


x = 100

print("Cosine Similarity 7 8:", cosine_similarity(X_1[1, [7], :], X_1[2, [7], :]))
print("Cosine Similarity 8 9:", cosine_similarity(X_1[x, [8], :], X_1[x, [9], :]))

avg = (X_1[x, [7], :] + X_1[x, [9], :] + X_1[x, [2], :] + X_1[x, [14], :]) / 4
print("Cosine Similarity 7 avg:", cosine_similarity(X_1[x, [7], :], avg))
print("pearsonr Similarity 7 8:", pearsonr(np.squeeze(avg), np.squeeze(X_1[x, [8], :]))[0])

avg = (X_1[x, [7], :] + X_1[x, [9], :]) / 2
print("Cosine Similarity 7 avg:", cosine_similarity(X_1[x, [7], :], avg))
print("pearsonr Similarity 7 8:", pearsonr(np.squeeze(avg), np.squeeze(X_1[x, [8], :]))[0])

print("=====================================")



# X_2 = X_2[:, :, 0:1001]
xx = 1245
print("Cosine Similarity 0 1:", cosine_similarity(X_2[xx, [0], :], X_2[xx, [2], :]))
# corr, _ = pearsonr(X_2[xx, [0], :], X_2[xx, [2], :])
print("pearsonr Similarity 7 8:", pearsonr(np.squeeze(X_2[xx, [0], :]), np.squeeze(X_2[xx, [1], :]))[0])
print("++++++++++++++++++++")
avg = (X_2[xx, [0], :] + X_2[xx, [2], :]) / 2
print("Cosine Similarity 0 1:", cosine_similarity(avg, X_2[xx, [2], :]))
print("pearsonr Similarity 7 8:", pearsonr(np.squeeze(avg), np.squeeze(X_2[xx, [2], :]))[0])




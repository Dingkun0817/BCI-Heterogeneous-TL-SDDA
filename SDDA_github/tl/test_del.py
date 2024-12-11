from sklearn.decomposition import PCA
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from imblearn.over_sampling import RandomOverSampler

xdcov = XdawnCovariances()
ts = TangentSpace()
train_cov = xdcov.fit_transform(Xt_train, Yt_train)
train_x_xdawn = ts.fit_transform(train_cov, Yt_train)

test_cov = xdcov.transform(Xt_test)
test_x_xdawn = ts.transform(test_cov)


# Random upsampling for balanced classification
train_x_xdawn, Yt_train = apply_randup(train_x_xdawn, Yt_train)

# PCA
train_x_xdawn, test_x_xdawn = apply_pca(train_x_xdawn, test_x_xdawn, 0.95)


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
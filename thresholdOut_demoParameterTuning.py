"""


This original paper from Dwork et al use feature selection to show off how well
thresholdout works. However, that's not a very useful application for big models
for two reasons:

    a) there's a 'budget' where you're only allowed to run threshold out
       a finite number of times. For high-dimensional datasets, you'll blow
       through that budget really quickly if you're running it on every feature.
       It's more frugal just to run it on model performance.

    b) It's difficult to measure feature importance in fancy models (ensemble methods
       of nonlinear models), so you won't be able to do the same feature selection.

"""



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

from scipy.stats import zscore
from sklearn.svm import SVC
from sklearn.datasets import make_classification


def createDataset(n, nPC):
    dim = int(np.log2(2 * nPC))
    X, Y = make_classification(n_samples=3 * n, n_classes=2,
                               n_clusters_per_class=nPC,
                               n_redundant=0, n_repeated=0,
                               n_features=4 * dim, n_informative=dim,
                               flip_y=0.0, class_sep=1.0)
    X_train = X[0::3, :]
    Y_train = Y[0::3]
    X_holdout = X[1::3, :]
    Y_holdout = Y[1::3]
    X_test = X[2::3, :]
    Y_test = Y[2::3]
    return [[X_train, Y_train], [X_holdout, Y_holdout], [X_test, Y_test]]


def createDataset_noSignal(n, nPC):
    dataset = createDataset(n, nPC)
    for d in dataset:
        np.random.shuffle(d[1])
    return dataset


def thresholdout(trParam, hoParam, scale):

    thr = scale
    tol = thr / 4

    trParam = np.array(trParam)
    hoParam = np.array(hoParam)
    newHoParam = np.copy(trParam)

    diffNoise = np.abs(trParam - hoParam) - np.random.normal(0, tol, hoParam.shape)
    flipIdx = diffNoise > thr
    newHoParam[flipIdx] = hoParam[flipIdx] + np.random.normal(0, tol, hoParam[flipIdx].shape)
    return newHoParam


def getBestGamma(data, model):
    bestGamma = 0.0
    bestAccy = -1.0
    for g in gammaList:
        model.set_params(gamma=g)
        model = model.fit(data[0][0], data[0][1])
        currAccy = model.score(data[1][0], data[1][1])
        if currAccy > bestAccy:
            bestAccy = currAccy
            bestGamma = g
    return bestGamma


def getModelPerf(data, model):
    model.set_params(gamma=getBestGamma(data, model))
    model = model.fit(data[0][0], data[0][1])
    scores = [model.score(d[0], d[1]) for d in data]
    return scores


def runClassifier(n, nPC, krange, nullData=False):

    if nullData:
        dataset = createDataset_noSignal(n, nPC)
    else:
        dataset = createDataset(n, nPC)

    model = SVC(C=1.0, kernel='rbf', gamma=0.5)

    # Get performance as a function of # of retained features
    vals = []
    vals_tho = []
    for k in krange:
        model.set_params(C=k)
        perf = getModelPerf(dataset, model)
        perf_tho = [perf[0],
                    thresholdout(perf[0], perf[1], 0.05),
                    perf[2]]
        vals.append(perf)
        vals_tho.append(perf_tho)

    return vals, vals_tho


def repeatexp(n, d, krange, reps, noSignal):
    """
    Repeat the experiment multiple times on different
    datasets to put errorbars on the graphs
    """
    condList = ['Train', 'Holdout', 'Test']
    colList = ['perm', 'paramVal', 'perf', 'dset']
    valDataframe_std = pd.DataFrame(data=[], columns=colList)
    valDataframe_tho = pd.DataFrame(data=[], columns=colList)
    for perm in range(reps):
        print("Repetition: {}".format(perm + 1))
        vals_std, vals_tho = runClassifier(n, d, krange, nullData=noSignal)
        tmpNew_std = []
        tmpNew_tho = []
        for i, cond in enumerate(condList):
            for j, paramVal in enumerate(krange):
                tmpNew_std.append([perm, paramVal, vals_std[j][i], cond])
                tmpNew_tho.append([perm, paramVal, vals_tho[j][i], cond])
        tmpNew_std = pd.DataFrame(tmpNew_std, columns=colList)
        tmpNew_tho = pd.DataFrame(tmpNew_tho, columns=colList)
        valDataframe_std = pd.concat([valDataframe_std, tmpNew_std], axis=0)
        valDataframe_tho = pd.concat([valDataframe_tho, tmpNew_tho], axis=0)
    return valDataframe_std, valDataframe_tho


def runandplotsummary(n, d, krange, reps):
    """
    Run the experiments with/without noise, put it in a subplot
    """

    print('Running on data WITH signal')
    df_std, df_tho = repeatexp(n, d, krange, reps, False)

    print('Running on data WITHOUT signal')
    df_std_ns, df_tho_ns = repeatexp(n, d, krange, reps, True)

    f, axes = plt.subplots(2, 2)
    sb.set_style('whitegrid')

    sb.tsplot(data=df_std,
              time='paramVal',
              unit='perm',
              condition='dset',
              value='perf',
              ax=axes[0][0])
    axes[0][0].set_title('Standard, HAS Signal')

    sb.tsplot(data=df_tho,
              time='paramVal',
              unit='perm',
              condition='dset',
              value='perf',
              ax=axes[0][1])
    axes[0][1].set_title('Thresholdout, HAS Signal')

    sb.tsplot(data=df_std_ns,
              time='paramVal',
              unit='perm',
              condition='dset',
              value='perf',
              ax=axes[1][0])
    axes[1][0].set_title('Standard, NO Signal')

    sb.tsplot(data=df_tho_ns,
              time='paramVal',
              unit='perm',
              condition='dset',
              value='perf',
              ax=axes[1][1])
    axes[1][1].set_title('Thresholdout, NO Signal')


#############################
#
# Run the experiment
#

"""
These two experiments will demonstrate how the use of thresholdout allows for
adaptive paramge tuning without overfitting to the trianing data.

num_per_condition is the number of clusters in each condition
krange is the range of 'gamma' parameters for a RBF SVM
"""

reps = 2
n, num_per_condition = 500, 4

gammaList = np.linspace(0, 4.0, 50 + 1)[1:]
krange    = np.linspace(0, 1.0, 20 + 1)[1:]


plt.close('all')
runandplotsummary(n, num_per_condition, krange, reps)

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

So here, we show how to use thresholdout in a parameter-tuning grid search that won't
overfit your models
"""



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

from scipy.stats import zscore
from sklearn.svm import SVC
from sklearn.datasets import make_classification


def createDataset(n, nPC, sd=1.0):
    centers = np.random.rand(1, 3, nPC * 2)
    tmpRand = np.random.randn((3 * n) // (nPC * 2), 3, nPC * 2) * sd
    X = tmpRand + centers
    Y = np.dstack([np.zeros((X.shape[0], 1, nPC)), np.ones((X.shape[0], 1, nPC))])
    X = X.reshape(-1, 3, order='F')
    Y = Y.reshape(-1, order='F')
    X_train = X[0::3, :]
    Y_train = Y[0::3]
    X_holdout = X[1::3, :]
    Y_holdout = Y[1::3]
    X_test = X[2::3, :]
    Y_test = Y[2::3]
    return [[X_train, Y_train], [X_holdout, Y_holdout], [X_test, Y_test]]


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


def getBestGamma(data, model, krange, gammaRange):
    bestGamma = 0.0
    bestC = 0.0
    bestGamma_tho = 0.0
    bestC_tho = 0.0
    bestAccy = -1.0
    bestAccy_tho = -1.0
    for g in gammaRange:
        for k in krange:
            model.set_params(C=k, gamma=g)
            model = model.fit(data[0][0], data[0][1])
            trAccy = model.score(data[0][0], data[0][1])
            currAccy = model.score(data[1][0], data[1][1])
            if currAccy > bestAccy:
                bestAccy = currAccy
                bestGamma = g
                bestC = k
            currAccy_tho = thresholdout(trAccy, currAccy, SCALE)
            if currAccy_tho > bestAccy_tho:
                bestAccy_tho = currAccy_tho
                bestGamma_tho = g
                bestC_tho = k
    return bestC, bestGamma, bestC_tho, bestGamma_tho


def getModelPerf(data, model, krange, gammaRange):
    bestParam = getBestGamma(data, model, krange, gammaRange)
    model.set_params(C=bestParam[0], gamma=bestParam[1])
    model = model.fit(data[0][0], data[0][1])
    scores = [model.score(d[0], d[1]) for d in data]

    model.set_params(C=bestParam[2], gamma=bestParam[3])
    model = model.fit(data[0][0], data[0][1])
    scores_tho = [model.score(d[0], d[1]) for d in data]
    return scores, scores_tho


def runClassifier(n, nPC, krange, gammaRange, nullData=False):

    dataset = createDataset(n, nPC, SIG_SD)
    if nullData:
        for d in dataset:
            np.random.shuffle(d[1])

    model = SVC(C=1.0, kernel='rbf', gamma=0.5)

    # Get performance as a function of free parameters
    vals = []
    vals_tho = []
    perf, perf_tho = getModelPerf(dataset, model, krange, gammaRange)
    vals.append(perf)
    vals_tho.append(perf_tho)
    return vals, vals_tho


def repeatexp(n, d, krange, gammaRange, reps, noSignal):
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
        vals_std, vals_tho = runClassifier(n, d, krange, gammaRange, nullData=noSignal)
        tmpNew_std = []
        tmpNew_tho = []
        for i, cond in enumerate(condList):
            tmpNew_std.append([perm, 0, vals_std[0][i], cond])
            tmpNew_tho.append([perm, 0, vals_tho[0][i], cond])
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
    df_std, df_tho = repeatexp(n, d, krange, gammaRange, reps, False)

    print('Running on data WITHOUT signal')
    df_std_ns, df_tho_ns = repeatexp(n, d, krange, gammaRange, reps, True)

    f, axes = plt.subplots(2, 2)
    sb.set_style('whitegrid')

    sb.barplot(data=df_std,
               x='dset',
               y='perf',
               ax=axes[0][0])
    axes[0][0].set_title('Standard, HAS Signal')

    sb.barplot(data=df_tho,
               x='dset',
               y='perf',
               ax=axes[0][1])
    axes[0][1].set_title('Thresholdout, HAS Signal')

    sb.barplot(data=df_std_ns,
               x='dset',
               y='perf',
               ax=axes[1][0])
    axes[1][0].set_title('Standard, NO Signal')

    sb.barplot(data=df_tho_ns,
               x='dset',
               y='perf',
               ax=axes[1][1])
    axes[1][1].set_title('Thresholdout, NO Signal')


#############################
#
# Run the experiment
#

"""
These two experiments will demonstrate how the use of thresholdout allows for
adaptive parameter tuning in a grid search without overfitting to the trianing data.

num_per_condition is the number of clusters in each condition
gammaRange is the range of 'gamma' parameters for a RBF SVM
krange is the range of soft-margin regularization parameters for the SVM, (e.g., C)

NOTE: if you have too much data realtive to the complexity of the problem, the
grid search won't overfit because the training and holdout data are each complete
enough to build the same models of the problem.

"""

reps = 100
n, num_per_condition = 50, 2

gammaRange = np.linspace(0, 4.0, 20 + 1)[1:]
krange = np.linspace(0, 1.0, 20 + 1)[1:]

SCALE = 0.20
SIG_SD = 0.5 / np.sqrt(num_per_condition)

plt.close('all')
runandplotsummary(n, num_per_condition, krange, reps)

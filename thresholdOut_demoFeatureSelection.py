"""
This script replicates the experiment from Dwork et al's original paper to show
how thresholdOut can be used for adaptive feature selection.

However, it's improved by allowing more complex (but still linear) classifiers
to work on the data rather than their simple correlation-based classifier.

ALSO, it allows for linear-regression!
"""




import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

from scipy.stats import zscore
from sklearn.linear_model import *


def createDataset(n, d, yBool=True):
    numInformativeFeatures = 20
    bias = 6.0 / np.sqrt(n)
    X = np.random.normal(0, 1, (3 * n, d + 1))

    if yBool:
        X[:, -1] = np.sign(X[:, -1])
        tmpIdx = X[:, -1] == 0
        X[tmpIdx, -1] = 1

    for i in range(numInformativeFeatures):
        X[:, i] += bias * X[:, -1]

    X_train = zscore(X[0::3, :-1], axis=0)
    Y_train = X[0::3, -1]
    X_holdout = zscore(X[1::3, :-1], axis=0)
    Y_holdout = X[1::3, -1]
    X_test = zscore(X[2::3, :-1], axis=0)
    Y_test = X[2::3, -1]
    return [[X_train, Y_train], [X_holdout, Y_holdout], [X_test, Y_test]]


def createDataset_noSignal(n, d, yBool=True):
    dataset = createDataset(n, d, yBool)
    for d in dataset:
        np.random.shuffle(d[1])
    return dataset


def getFeatureValues(trFeat, hoFeat):
    cutoff_tr = 1.0 * np.sqrt(np.mean(trFeat**2))  # 1 / np.sqrt(n)
    cutoff_ho = 1.0 * np.sqrt(np.mean(hoFeat**2))  # 1 / np.sqrt(n)
    bothPos = (trFeat > cutoff_tr) & (hoFeat > cutoff_ho)
    bothNeg = (trFeat < -cutoff_tr) & (hoFeat < -cutoff_ho)
    goodFeat = bothPos | bothNeg
    featValue = np.copy(np.abs(trFeat).ravel())
    featValue[~goodFeat] = 0.0
    return featValue


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


def getCoeffs(data, model):
    coeff = model.fit(data[0], data[1]).coef_.ravel()
    return coeff


def getModelPerf(useFeat, data, model):
    useFeat = useFeat.ravel()
    if useFeat.shape[0] > 0:
        model = model.fit(data[0][0][:, useFeat], data[0][1])
        scores = [model.score(d[0][:, useFeat], d[1]) for d in data]
    else:
        if model._estimator_type == 'regressor':
            scores = [0.0, 0.0, 0.0]
        elif model._estimator_type == 'classifier':
            scores = [0.5, 0.5, 0.5]
        else:
            scores = [-999, -999, -999]
    return scores


def runClassifier(n, d, krange, nullData=False):

    if nullData:
        dataset = createDataset_noSignal(n, d, yBool=True)
    else:
        dataset = createDataset(n, d, yBool=True)

    if isClsf:
        model = LogisticRegression(penalty='l2', C=1.0)
    else:
        model = LinearRegression()

    # Get feature coefficients for training, holdout (actual), and holdout (with THO)
    trainCoeff = getCoeffs(dataset[0], model)
    holdoutCoeff = getCoeffs(dataset[1], model)
    holdoutCoeff_tho = thresholdout(trainCoeff, holdoutCoeff, 1.0)

    # Use an 'adaptive' approach to finding strong features
    featureValue = getFeatureValues(trainCoeff, holdoutCoeff)
    featureValue_tho = getFeatureValues(trainCoeff, holdoutCoeff_tho)

    # Rank the goodness of each feature
    featureRank = np.argsort(-featureValue)
    featureRank_tho = np.argsort(-featureValue_tho)

    # Get performance as a function of # of retained features
    vals = [getModelPerf(featureRank[:k], dataset, model) for k in krange]
    vals_tho = [getModelPerf(featureRank_tho[:k], dataset, model) for k in krange]
    # for v in vals_tho:
    #    v[1] = thresholdout(v[0], v[1], 0.02)

    return vals, vals_tho


def repeatexp(n, d, krange, reps, noSignal):
    """
    Repeat the experiment multiple times on different
    datasets to put errorbars on the graphs
    """
    condList = ['Train', 'Holdout', 'Test']
    colList = ['perm', 'numFeat', 'perf', 'dset']
    valDataframe_std = pd.DataFrame(data=[], columns=colList)
    valDataframe_tho = pd.DataFrame(data=[], columns=colList)
    for perm in range(reps):
        print("Repetition: {}".format(perm + 1))
        vals_std, vals_tho = runClassifier(n, d, krange, nullData=noSignal)
        for i, cond in enumerate(condList):
            for j, numFeat in enumerate(krange):
                tmpNew_std = pd.DataFrame([[perm, numFeat, vals_std[j][i], cond]], columns=colList)
                tmpNew_tho = pd.DataFrame([[perm, numFeat, vals_tho[j][i], cond]], columns=colList)
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
              time='numFeat',
              unit='perm',
              condition='dset',
              value='perf',
              ax=axes[0][0])
    axes[0][0].set_title('Standard, HAS Signal')

    sb.tsplot(data=df_tho,
              time='numFeat',
              unit='perm',
              condition='dset',
              value='perf',
              ax=axes[0][1])
    axes[0][1].set_title('Thresholdout, HAS Signal')

    sb.tsplot(data=df_std_ns,
              time='numFeat',
              unit='perm',
              condition='dset',
              value='perf',
              ax=axes[1][0])
    axes[1][0].set_title('Standard, NO Signal')

    sb.tsplot(data=df_tho_ns,
              time='numFeat',
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
adaptive feature selection without overfitting to the trianing data.

isClsf=True will demonstrate using logistic regression classifier
isClsf=False will demonstrate using multiple regression

Note: The benefits of thresholdOut become clearer as 'd' gets bigger
"""

reps = 20
n, d = 1000, 10000
krange = list(range(0, 40, 2))

isClsf = True

plt.close('all')
runandplotsummary(n, d, krange, reps)

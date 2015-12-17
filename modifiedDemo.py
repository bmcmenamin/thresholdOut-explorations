import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

from scipy.stats import zscore
from sklearn.linear_model import *


def createnosignaldata(n, d, yBool=True):
    X = np.random.normal(0, 1, (3 * n, d + 1))
    if yBool:
        X[:, -1] = np.sign(X[:, -1])
        tmpIdx = X[:, -1] == 0
        X[tmpIdx, -1] = 1
    return [X[0::3, :], X[1::3, :], X[2::3, :]]


def createhighsignaldata(n, d, yBool=True):
    numInformativeFeatures = 20
    bias = 6.0 / np.sqrt(n)
    X = np.random.normal(0, 1, (3 * n, d + 1))
    if yBool:
        X[:, -1] = np.sign(X[:, -1])
        tmpIdx = X[:, -1] == 0
        X[tmpIdx, -1] = 1
    for i in range(numInformativeFeatures):
        X[:, i] += bias * X[:, -1]
    return [X[0::3, :], X[1::3, :], X[2::3, :]]


def adaptiveFeatureSelection(feat1, feat2):
    cutoff = 1 / np.sqrt(n)
    bothPos = (feat1 > cutoff) & (feat2 > cutoff)
    bothNeg = (feat1 < -cutoff) & (feat2 < -cutoff)
    goodFeat = bothPos | bothNeg
    featVal = np.copy(feat1)
    featVal[~goodFeat] = 0.0
    return featVal



def thresholdout(train, holdout, thr, tol):
    diff = np.abs(train - holdout)
    diffNoise = diff - np.random.normal(0, tol, diff.shape)
    flipIdx = diffNoise > thr
    noise = np.random.normal(0, tol, train[flipIdx].shape)
    newVec = np.copy(train)
    newVec[flipIdx] = holdout[flipIdx] + noise
    return newVec


def getModelCoeffs(X, model=None):
    if model:
        coeff = model.fit(X[:, :-1], X[:, -1]).coef_.ravel()
    else:
        coeff = X[:, :-1].T.dot(X[:, -1:]) / X.shape[0]
    return coeff


def getModelPerf(useFeat, X_train, X_holdout, X_test, model=None):
    useFeat = useFeat.ravel()
    if model:
        model = model.fit(X_train[:, useFeat], X_train[:, -1])
        ftrain = model.score(X_train[:, useFeat], X_train[:, -1])
        fholdout = model.score(X_holdout[:, useFeat], X_holdout[:, -1])
        ftest = model.score(X_test[:, useFeat], X_test[:, -1])
    else:
        w = np.sign(getModelCoeffs(X_train)[useFeat])
        ftrain = np.mean(np.sign(X_train[:, useFeat].dot(w)) == X_train[:, -1:])
        fholdout = np.mean(np.sign(X_holdout[:, useFeat].dot(w)) == X_holdout[:, -1:])
        ftest = np.mean(np.sign(X_test[:, useFeat].dot(w)) == X_test[:, -1:])
    return ftrain, fholdout, ftest


def runClassifier(n, d, krange, X_train, X_holdout, X_test):
    if isClsf:
        model = LogisticRegression(penalty='l2', C=1.0)
        X_train[:, :-1] = zscore(X_train[:, :-1], axis=0)
        X_holdout[:, :-1] = zscore(X_holdout[:, :-1], axis=0)
        X_test[:, :-1] = zscore(X_test[:, :-1], axis=0)
    else:
        model = LinearRegression()
        X_train = zscore(X_train, axis=0)
        X_holdout = zscore(X_holdout, axis=0)
        X_test = zscore(X_test, axis=0)

    trainCoeff = getModelCoeffs(X_train, model=None)
    holdoutCoeff = getModelCoeffs(X_holdout, model=None)
    holdoutCoeff_noisy = thresholdout(trainCoeff, holdoutCoeff, threshold, tolerance)

    featureValue = adaptiveFeatureSelection(trainCoeff, holdoutCoeff).ravel()
    featureValue_noisy = adaptiveFeatureSelection(trainCoeff, holdoutCoeff_noisy).ravel()
    featureRank = np.argsort(-np.abs(featureValue))
    featureRank_noisy = np.argsort(-np.abs(featureValue_noisy))

    vals = []
    noisy_vals = []
    for k in krange:
        if k == 0:
            if isClsf:
                vals.append([0.5, 0.5, 0.5])
                noisy_vals.append([0.5, 0.5, 0.5])
            else:
                vals.append([0.0, 0.0, 0.0])
                noisy_vals.append([0.0, 0.0, 0.0])
        else:
            tmpOut = list(getModelPerf(featureRank[:k], X_train, X_holdout, X_test, model=None))
            vals.append(tmpOut)

            tmpOut = list(getModelPerf(featureRank_noisy[:k], X_train, X_holdout, X_test, model=None))
            # tmpOut[1] = thresholdout(np.array(tmpOut[0]), np.array(tmpOut[1]), threshold, tolerance)
            noisy_vals.append(tmpOut)
    return vals, noisy_vals


def repeatexp(n, d, krange, reps, datafn):
    """ Repeat experiment specified by fn for reps steps """
    condList = ['Train', 'Holdout', 'Test']
    colList = ['perm', 'numFeat', 'perf', 'dset']
    valDataframe1 = pd.DataFrame(data=[], columns=colList)
    valDataframe2 = pd.DataFrame(data=[], columns=colList)
    for perm in range(reps):
        print("Repetition: {}".format(perm))
        dataSets = datafn(n, d, yBool=isClsf)
        vals, vals2 = runClassifier(n, d, krange, *dataSets)
        for i, cond in enumerate(condList):
            for j, numFeat in enumerate(krange):
                tmpNew1 = pd.DataFrame([[perm, numFeat, vals[j][i], cond]], columns=colList)
                tmpNew2 = pd.DataFrame([[perm, numFeat, vals2[j][i], cond]], columns=colList)
                valDataframe1 = pd.concat([valDataframe1, tmpNew1], axis=0)
                valDataframe2 = pd.concat([valDataframe2, tmpNew2], axis=0)
    return valDataframe1, valDataframe2


def runandplotsummary(n, d, krange, reps, datafn):
    df1, df2 = repeatexp(n, d, krange, reps, datafn)
    plt.figure()
    sb.set_style('whitegrid')
    sb.tsplot(data=df1,
              time='numFeat',
              unit='perm',
              condition='dset',
              value='perf')
    plt.title('Standard')
    plt.figure()
    sb.set_style('whitegrid')
    sb.tsplot(data=df2,
              time='numFeat',
              unit='perm',
              condition='dset',
              value='perf')
    plt.title('Thresholdout')







reps = 20
n, d = 5000, 500
krange = list(range(0, 100, 4))

threshold = 4.0 / np.sqrt(n)
tolerance = threshold / 4.0

isClsf = True

# Experiment 1:
# No correlations
runandplotsummary(n, d, krange, reps, createnosignaldata)

# Experiment 2:
# Some variables are correlated
runandplotsummary(n, d, krange, reps, createhighsignaldata)



datasets = createhighsignaldata(n, d, yBool=True)
X_train = datasets[0]
X_holdout = datasets[1]
X_test = datasets[2]

getModelCoeffs(X_train, model=None)

import numpy as np
import datetime
import matplotlib.pyplot as plt

def createnosignaldata(n, d):
    """
    Data points are random Gaussian vectors.
    Class labels are random and uniform
    """
    
    X_train = np.random.normal(0, 1, (n, d + 1))
    X_train[:, d] = np.sign(X_train[:, d])
    X_holdout = np.random.normal(0, 1, (n, d + 1))
    X_holdout[:, d] = np.sign(X_holdout[:, d])
    X_test = np.random.normal(0, 1, (n, d + 1))
    X_test[:, d] = np.sign(X_test[:, d])
    
    return X_train, X_holdout, X_test


def createhighsignaldata(n, d):
    """
    Data points are random Gaussian vectors.
    Class labels are random and uniform
    First nbiased are biased with bias towards the class label
    """
    
    X_train = np.random.normal(0, 1, (n, d + 1))
    X_train[:, d] = np.sign(X_train[:, d])
    X_holdout = np.random.normal(0, 1, (n, d + 1))
    X_holdout[:, d] = np.sign(X_holdout[:, d])
    X_test = np.random.normal(0, 1, (n, d + 1))
    X_test[:, d] = np.sign(X_test[:, d])

    # Add correlation with the sign
    nbiased = 20
    bias = 6.0 / np.sqrt(n)
    b = np.zeros(nbiased)
    for i in range(n):
        b[:nbiased] = bias * X_holdout[i, d]
        X_holdout[i, :nbiased] = np.add(X_holdout[i, :nbiased], b)
        b[:nbiased] = bias * X_train[i, d]
        X_train[i, :nbiased] = np.add(X_train[i, :nbiased], b)
        b[:nbiased] = bias * X_test[i, d]
        X_test[i, :nbiased] = np.add(X_test[i, :nbiased], b)
    return X_train, X_holdout, X_test


def runClassifier(n, d, krange, X_train, X_holdout, X_test):
    
    """
    Variable selection and basic boosting on synthetic data. Variables
    with largest correlation with target are selected first.
    """
    
    # Compute values on the standard holdout
    tolerance = 1.0 / np.sqrt(n)
    threshold = 4.0 / np.sqrt(n)

    vals = []
    trainanswers = np.dot(X_train[:, :d].T, X_train[:, d]) / n
    holdoutanswers = np.dot(X_holdout[:, :d].T, X_holdout[:, d]) / n
    trainpos = trainanswers > 1.0 / np.sqrt(n)
    holdopos = holdoutanswers > 1.0 / np.sqrt(n)
    trainneg = trainanswers < -1.0 / np.sqrt(n)
    holdoneg = holdoutanswers < -1.0 / np.sqrt(n)
    selected = (trainpos & holdopos) | (trainneg & holdoneg)
    trainanswers[~selected] = 0
    sortanswers = np.abs(trainanswers).argsort()
    for k in krange:
        weights = np.zeros(d + 1)
        topk = sortanswers[-k:]
        weights[topk] = np.sign(trainanswers[topk])
        ftrain = 1.0 * np.count_nonzero(np.sign(np.dot(X_train, weights)) == X_train[:, d]) / n
        fholdout = 1.0 * np.count_nonzero(np.sign(np.dot(X_holdout, weights)) == X_holdout[:, d]) / n
        ftest = 1.0 * np.count_nonzero(np.sign(np.dot(X_test, weights)) == X_test[:, d]) / n
        if k == 0:
            vals.append([0.5, 0.5, 0.5])
        else:
            vals.append([ftrain, fholdout, ftest])

    # Compute values using Thresholdout
    noisy_vals = []
    trainanswers = np.dot(X_train[:, :d].T, X_train[:, d]) / n
    holdoutanswers = np.dot(X_holdout[:, :d].T, X_holdout[:, d]) / n
    diffs = np.abs(trainanswers - holdoutanswers)
    noise = np.random.normal(0, tolerance, d)
    abovethr = diffs > threshold + noise
    holdoutanswers[~abovethr] = trainanswers[~abovethr]
    holdoutanswers[abovethr] = (holdoutanswers + np.random.normal(0, tolerance, d))[abovethr]
    trainpos = trainanswers > 1.0 / np.sqrt(n)
    holdopos = holdoutanswers > 1.0 / np.sqrt(n)
    trainneg = trainanswers < -1.0 / np.sqrt(n)
    holdoneg = holdoutanswers < -1.0 / np.sqrt(n)
    selected = (trainpos & holdopos) | (trainneg & holdoneg)
    trainanswers[~selected] = 0
    sortanswers = np.abs(trainanswers).argsort()
    for k in krange:
        weights = np.zeros(d + 1)
        topk = sortanswers[-k:]
        weights[topk] = np.sign(trainanswers[topk])
        ftrain = 1.0 * np.count_nonzero(np.sign(np.dot(X_train, weights)) == X_train[:, d]) / n
        fholdout = 1.0 * np.count_nonzero(np.sign(np.dot(X_holdout, weights)) == X_holdout[:, d]) / n
        if abs(ftrain - fholdout) < threshold + np.random.normal(0, tolerance):
            fholdout = ftrain
        else:
            fholdout += np.random.normal(0, tolerance)
        ftest = 1.0 * np.count_nonzero(np.sign(np.dot(X_test, weights)) == X_test[:, d]) / n
        if k == 0:
            noisy_vals.append([0.5, 0.5, 0.5])
        else:
            noisy_vals.append([ftrain, fholdout, ftest])

    vals = np.array(vals)
    noisy_vals = np.array(noisy_vals)
    return vals, noisy_vals

def repeatexp(n, d, krange, reps, datafn):
    """ Repeat experiment specified by fn for reps steps """
    vallist = []
    vallist_noisy = []
    for r in range(0, reps):
        #print("Repetition: {}".format(r))
        X_train, X_holdout, X_test = datafn(n, d)
        vals, vals_noisy = runClassifier(n, d, krange, X_train, X_holdout, X_test)
        vallist.append(vals)
        vallist_noisy.append(vals_noisy)
 
    vallist = np.dstack(vallist)
    vallist_noisy = np.dstack(vallist_noisy)
    
    return vallist, vallist_noisy


def runandplotsummary(n, d, krange, reps, datafn, condName):
    
    vallist_normal, vallist_tout = repeatexp(n, d, krange, reps, datafn)
    
    mean_normal = np.mean(vallist_normal, axis=2)
    std_normal  = np.std(vallist_normal, axis=2)
    
    mean_tout = np.mean(vallist_tout, axis=2)
    std_tout  = np.std(vallist_tout, axis=2)

    ts = datetime.datetime.now().strftime('%Y%m%d%H%M')
    plotname = "plot-{ts}-{n}-{d}-{reps}-{condition}"
    plotname = plotname.format(ts=ts, n=n, d=d, reps=reps, condition=condName)

    f, ax = plt.subplots(2, 1, sharex=True)
    plot1(ax[0], krange, mean_normal, std_normal, plotname + "-std", "Standard holdout")
    plot1(ax[1], krange, mean_tout,   std_tout,   plotname + "-thr", "Thresholdout")
    ax[1].set_xlabel('Number of variables', fontsize='8')
    ax[1].legend(loc=2, prop={'size': 6})    
 
def plot1(a, x, m, sd, plotname, plottitle):
    
    a.set_title(plottitle, fontsize='8')
    a.set_ylabel('Accuracy', fontsize='8')
    a.axis([x[0], x[-1], 0.45, 0.75])

    colorList = ['#B2B2F5', '#CCFFCC', '#FF9848']
    label = ['training', 'holdout', 'fresh']
    for i, color in enumerate(colorList):
        a.plot(x, m[:, i], c=color, marker='^', label=label[i])
        a.fill_between(x, m[:, i] - sd[:, i], m[:, i] + sd[:, i],
                       alpha=0.5, edgecolor=color, facecolor=color, linestyle='dashdot')
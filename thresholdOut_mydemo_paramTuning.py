import numpy as np
import pandas as pd

from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression, LassoLars

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sb

def createDataset(n, d=100, d_inf=5, is_classification=True, no_signal=False):
    """
    n is number of samples in each of train/test/holdout sets
    d is the data dimension (default=100)
    d_inf is the number of informative features (default=20)
    is_classification is bool stating whether data is for classification
        as opposed to regression (default=True)
    no_signal is a bool stating whether you want the data randomized so
        there's no useful signal (default=False)
    """
    
    # making random inputs, outputs
    X = np.random.normal(0, 1, (3*n, d))
    y = np.random.normal(0, 1, (3*n, 1))
    
    # thresholding y values for classification
    if is_classification:
        y = 2.0*((y>0) - 0.5)
    
    # making the first d_inf dimensions informative

    if is_classification:
        X[:,:d_inf] += y*np.random.normal(1.0, 1.5, X[:,:d_inf].shape)
    else:
        snr = 0.05
        X[:,:d_inf] += snr*y
    X = zscore(X, axis=0)
    
    # if you dont want useful signal, randomize the labels
    if no_signal:
        np.random.shuffle(y)

    # Divide into train/test/holdout pairs
    outputs = [[X[i::3, :], y[i::3, 0]] for i in range(3)]    
    
    return outputs

def thresholdout(train_vals, holdout_vals, tho_scale=1.0):
    
    """
    This is the actual thresholdout algorithm
    that takes values from a training-run and a holdout-run
    and returns a new set of holdout values
    """
    
    thr = tho_scale
    tol = thr / 4

    train_vals = np.array(train_vals)
    holdout_vals = np.array(holdout_vals)
    
    diffNoise = np.abs(train_vals - holdout_vals) - np.random.normal(0, tol, holdout_vals.shape)
    flipIdx = diffNoise > thr
    
    new_holdout_vals = np.copy(train_vals)
    new_holdout_vals[flipIdx] = np.copy(holdout_vals)[flipIdx] + np.random.normal(0, tol, new_holdout_vals[flipIdx].shape)
    return new_holdout_vals


def fitModels_paramTuning(n, d, grid_size, no_signal=False, tho_scale=0.1, is_classification=True):

    dataset = createDataset(n, d=d, d_inf=10,
                            is_classification=True,
                            no_signal=no_signal)
    
    best_perf_std = [-np.inf, -np.inf, -np.inf]
    best_perf_tho = [-np.inf, -np.inf, -np.inf]
    best_tho = -np.inf
    for a in np.logspace(-4, 0, num=grid_size):
        if is_classification:
            model = LogisticRegression(penalty='l1', C=a)
            eval_model = lambda x: model.score(*x)
        else:
            model = LassoLars(alpha=a, normalize=True, max_iter=5000)
            #eval_model = lambda x: (np.var(x[1]) - np.mean((model.predict(x[0]) - x[1])**2)) / np.var(x[1])
            eval_model = lambda x: model.score(*x)
        
        # Train models
        model = model.fit(*dataset[0])
        
        # standard holdout performance
        perf_std = [eval_model(d) for d in dataset]        
        if perf_std[1] > best_perf_std[1]:
            best_perf_std = perf_std
        
        # thresholdout performance
        _tho = thresholdout(perf_std[0], perf_std[1], tho_scale=tho_scale)
        perf_tho = [i for i in perf_std]
        if _tho > best_tho:
            best_tho = _tho
            best_perf_tho = perf_tho
            
    return best_perf_std, best_perf_tho

def repeatexp(n, d, grid_size, reps, tho_scale=0.1, is_classification=True, no_signal=True):
    """
    Repeat the experiment multiple times on different
    datasets to put errorbars on the graphs
    """
    
    datasetList = ['Train', 'Holdout', 'Test']
    colList = ['perm', 'performance', 'dataset']
    
    df_list_std = []
    df_list_tho = []
    
    for perm in tqdm(range(reps)):
        
        vals_std, vals_tho = fitModels_paramTuning(n, d, grid_size,
                                                   is_classification=is_classification,
                                                   tho_scale=tho_scale,
                                                   no_signal=no_signal)
        for i, ds in enumerate(datasetList):
            df_list_std.append((perm, vals_std[i], ds))
            df_list_tho.append((perm, vals_tho[i], ds))

    df_std = pd.DataFrame(df_list_std, columns=colList)
    df_tho = pd.DataFrame(df_list_tho, columns=colList)
    return df_std, df_tho


def runExpt_and_makePlots(n, d, grid_size, reps, tho_scale=0.1, is_classification=True):
    """
    Run the experiments with and without useful training signal
    then make subplots to show how overfitting differs for
    standard holdout and thresholdout
    
    n = number of training samples in train/test/holdout sets
    d = dimension of data
    grid_size = number of steps in parameter grid search
    reps = number of times experiment is repeated
    is_classification = bool that indicates whether to do classification or regression
    """

    args = [n, d, grid_size, reps]
    df_std_signal, df_tho_signal     = repeatexp(*args,
                                                 is_classification=is_classification,
                                                 tho_scale=tho_scale,
                                                 no_signal=False)
    
    df_std_nosignal, df_tho_nosignal = repeatexp(*args,
                                                 is_classification=is_classification,
                                                 tho_scale=tho_scale,
                                                 no_signal=True)

    f, ax = plt.subplots(2, 2, figsize=(8,10), sharex=True, sharey=False)
    sb.set_style('whitegrid')
    
    kw_params = {'x':'dataset',
                 'y':'performance',
                 'units':'perm'}
    
    sb.barplot(data=df_std_signal,
               ax=ax[0,0],
               **kw_params)
    ax[0,0].set_title('Standard, HAS Signal')
    
    sb.barplot(data=df_tho_signal,
               ax=ax[0,1],
               **kw_params)
    ax[0,1].set_title('Thresholdout, HAS Signal')

    sb.barplot(data=df_std_nosignal,
               ax=ax[1,0],
               **kw_params)
    ax[1,0].set_title('Standard, NO Signal')

    sb.barplot(data=df_tho_nosignal,
               ax=ax[1,1],
               **kw_params)
    ax[1,1].set_title('Thresholdout, NO Signal')
    
    return f, ax
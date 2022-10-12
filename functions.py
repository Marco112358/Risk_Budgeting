import numpy as np


def exp_ma_std_fnc(arr=None, half_life=0.0, past2present=True):
    # parameters: arr = array of data to get ema and exp_std;
    # half_life is the half life of the exponential you want to use, period is in the period of the time series
    # for example: if you put in 100 days of data, and you want half of the weight to come from the 1st 20 days, put 20
    # past2present says True if your data is in chronological order
    n = arr.shape[0]

    if half_life == 0.0:
        decay = (n - 1) / (n + 1)
    else:
        decay = (0.5) ** (1 / half_life)
    if past2present == False:
        arng = np.arange(1, n + 1)
    else:
        arng = np.arange(n + 1, 1, step=-1)

    wghts = decay ** arng
    ewma = arr * wghts
    ema = np.sum(ewma) / np.sum(wghts)
    evar = ((arr - ema) ** 2) * wghts
    exp_std = (np.sum(evar) / np.sum(wghts)) ** 0.5

    return ema, exp_std

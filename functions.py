import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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


def get_prc_expo_moments(df=None, lookback=100, halflife=0.0, col_nm='na'):
    data = df.tail(lookback)
    arry = np.array(data.loc[:, col_nm])
    prc_ema, prc_estd = exp_ma_std_fnc(arr=arry, half_life=halflife, past2present=True)
    return prc_ema, prc_estd


def get_rtn_expo_moments(df=None, lookback=100, halflife=0.0, col_nm='na'):
    data = df.tail(lookback+1).reset_index()
    s1 = data[0:lookback]
    s2 = data[1:lookback+1]
    rtns = np.array(s2.loc[:, col_nm]) / np.array(s1.loc[:, col_nm]) - 1
    rtn_ema, rtn_estd = exp_ma_std_fnc(arr=rtns, half_life=halflife, past2present=True)
    return rtn_ema, rtn_estd


def get_simple_moments(df=None, lookback=100,  col_nm='na'):
    data = df.tail(lookback)
    arry = np.array(data.loc[:, col_nm])
    prc_ave = np.average(arry)
    prc_std = np.std(arry)
    return prc_ave, prc_std


def get_rtn_simple_moments(df=None, lookback=100,  col_nm='na'):
    data = df.tail(lookback+1).reset_index()
    s1 = data[0:lookback]
    s2 = data[1:lookback+1]
    rtns = np.array(s2.loc[:, col_nm]) / np.array(s1.loc[:, col_nm]) - 1
    rtn_ave = np.average(rtns)
    rtn_std = np.std(rtns)
    return rtn_ave, rtn_std


def rebal_by_period(timeperiod=100, rebal_freq=None, prices_final=None, st_dollars=10000, tgt_wghts=None, fee_pct=0.0):
    st_vals = np.multiply(st_dollars, tgt_wghts)
    st_prcs = prices_final.head(n=1)
    st_tkns = st_vals / st_prcs
    tkns_final = pd.DataFrame(index=prices_final.index, columns=prices_final.columns)
    fees = pd.DataFrame(data=0, index=prices_final.index, columns=['Fees'])
    tkns_final.iloc[0] = st_tkns
    rebals = np.arange(rebal_freq, timeperiod, step=rebal_freq)
    for i in np.arange(1, timeperiod):
        if i in rebals:
            prcs = prices_final.iloc[i]
            act_vals = tkns_final.iloc[i - 1] * prcs
            pv_prefee = np.sum(act_vals)
            tgt_vals_prefee = np.multiply(pv_prefee, tgt_wghts)
            diffs = act_vals - tgt_vals_prefee
            total_trade_val = np.sum(np.abs(diffs))
            fee = total_trade_val * fee_pct
            fees.iloc[i] = fee
            pv_postfee = pv_prefee - fee
            tgt_vals_postfee = np.multiply(pv_postfee, tgt_wghts)
            tkns_final.iloc[i] = tgt_vals_postfee / prcs
        else:
            tkns_final.iloc[i] = tkns_final.iloc[i - 1]
    return tkns_final, fees

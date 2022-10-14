import numpy as np

from functions import exp_ma_std_fnc


def exp_ma_std_fnc_old(arr=None, half_life=0.0, past2present=True):
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


def get_rtn_expo_moments(df=None, lookback=100, halflife=0.0, col_nm='na'):
    data = df.tail(lookback+1).reset_index()
    s1 = data[0:lookback]
    s2 = data[1:lookback+1]
    # rtns = np.array(s2.loc[:, col_nm]) / np.array(s1.loc[:, col_nm]) - 1
    rtns = (s2.loc[:, col_nm]) / np.array(s1.loc[:, col_nm]) - 1
    rtn_ema, rtn_estd = exp_ma_std_fnc(df=rtns, half_life=halflife)
    # rtn_ema, rtn_estd = exp_ma_std_fnc(arr=rtns, half_life=halflife, past2present=True)
    return rtn_ema, rtn_estd


def get_prc_expo_moments(df=None, lookback=100, halflife=0.0, col_nm='na'):
    data = df.tail(lookback)
    # arr = np.array(data.loc[:, col_nm])
    df_in = data.loc[:, col_nm]
    # prc_ema, prc_estd = exp_ma_std_fnc(arr=arry, half_life=halflife, past2present=True)
    prc_ema, prc_estd = exp_ma_std_fnc(df=df_in, half_life=halflife)
    return prc_ema, prc_estd


def get_rtn_simple_moments(df=None, lookback=100,  col_nm='na'):
    data = df.tail(lookback+1).reset_index()
    s1 = data[0:lookback]
    s2 = data[1:lookback+1]
    rtns = np.array(s2.loc[:, col_nm]) / np.array(s1.loc[:, col_nm]) - 1
    rtn_ave = np.average(rtns)
    rtn_std = np.std(rtns)
    return rtn_ave, rtn_std



asset_sds = np.matrix([0.15, 0.2, 0.05])  # Asset standard deviations
asset_corrs = np.matrix([[1.0, 0.8, 0.2], [0.8, 1.0, -0.1], [0.2, -0.1, 1.0]])  # Correlation Matrix
b = np.matrix([0.4, 0.2, 0.4])  # target risk contributions (PCTRs)
std_tgt = 0.1  # target ex-ante standard deviation
asset_covars = np.multiply(asset_sds.transpose(), np.multiply(asset_corrs, asset_sds))
# f_total = lambda x: 0.5 * np.matmul(x, np.matmul(asset_covars, x.transpose()))[0, 0] \
#               - np.matmul(b, np.log(x).transpose())[0, 0]

# Functions used in newton algo
fn = lambda x: np.matmul(asset_covars, x.transpose()) - (b / x).transpose()
f_prime = lambda x: asset_covars + np.diagflat((b / np.multiply(x, x)))


# Newton Algo
def my_newton(f, df, x0, tol, iter_tot):
    iter_n = 0
    if np.sum(abs(f(x0))) <= tol:
        return x0, iter_n
    else:
        x_new = x0 - np.matmul(np.linalg.inv(df(x0)), f(x0)).transpose()
        for i in np.arange(0, iter_tot):
            if np.sum(abs(f(x_new))) > tol:
                x_new = x_new - np.matmul(np.linalg.inv(df(x_new)), f(x_new)).transpose()
                iter_n = i
            else:
                return x_new, iter_n
        return x_new, iter_n


# Initial Guess for Algo
x0_test = np.sum(asset_sds, 1)[0, 0] / b

# Run Algo, get weights, print
x_out, itn = my_newton(fn, f_prime, x0_test, 0.000001, 10000)
wghts_out = x_out / np.matmul(np.matrix(np.ones(x_out.shape[1])), x_out.transpose())
print(wghts_out)
print(itn)
port_sd = np.sqrt(np.matmul(wghts_out, np.matmul(asset_covars, wghts_out.transpose())))[0, 0]
mctrs = np.matmul(asset_covars, wghts_out.transpose()) / port_sd
ctrs = np.multiply(mctrs, wghts_out.transpose())
pctrs = ctrs / port_sd
print(port_sd)
print(pctrs)

# Levering the Portfolio Weights to get to the targeted total portfolio standard deviation level
lev = std_tgt / port_sd
new_wghts = lev * wghts_out
new_port_sd = np.sqrt(np.matmul(new_wghts, np.matmul(asset_covars, new_wghts.transpose())))[0, 0]
new_mctrs = np.matmul(asset_covars, new_wghts.transpose()) / new_port_sd
new_ctrs = np.multiply(new_mctrs, new_wghts.transpose())
new_pctrs = new_ctrs / new_port_sd
print(new_wghts)
print(new_port_sd)
print(new_pctrs)

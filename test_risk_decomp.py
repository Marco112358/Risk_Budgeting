from api_testing import cg_pull
import functions as fn
import pandas as pd
import functools as ft
import numpy as np
import statsmodels.api as sm

# Parameters of Risk Balancing Backtesting
token_nms = ['btc', 'atom', 'sol']
col_nms = ['datetime'] + token_nms
start_dt = pd.Timestamp(year=2021, month=12, day=31)  # start really on the day following
st_dollars = 10000
fee_pct = 0.003
tgt_wghts = [0.33333333, 0.33333333, 0.33333333]

weighting_type = 'exp'  # options are exp for exponential or arth for arithmetic
tol = 0.00001
iter_tot = 10000
int_paid = 0.02
int_rec = int_paid

# API Pulls from Coin Gecko
btc = cg_pull('bitcoin', 'usd', 'max', 'daily')
atom = cg_pull('cosmos', 'usd', 'max', 'daily')
kuji = cg_pull('kujira', 'usd', 'max', 'daily')
usdc = cg_pull('usd-coin', 'usd', 'max', 'daily')
eth = cg_pull('ethereum', 'usd', 'max', 'daily')
sol = cg_pull('solana', 'usd', 'max', 'daily')
ada = cg_pull('cardano', 'usd', 'max', 'daily')
dfs = [btc, atom, sol]  # this needs to be changed if u change the tokens allocated to

# Implied Parameters
prices = ft.reduce(lambda left, right: pd.merge(left, right, on='datetime'), dfs)
prices.columns = col_nms
prices.set_index('datetime', inplace=True)
prices_final = prices.loc[prices.index >= start_dt]
rtns_final = np.divide(prices_final.iloc[1:], prices_final.iloc[:-1]) - 1

timeperiod = prices_final.shape[0]
rtn_timeperiod = rtns_final.shape[0]

ave, std, covar, corr = fn.get_simple_moments(rtns_final, rtn_timeperiod)
ave_ann, std_ann, covar_ann = fn.annualize_moments(ave, std, covar, 1)
pctrs = fn.pctrs(asset_sds=std_ann, asset_corrs=None, wghts=np.array(tgt_wghts), asset_covars=covar_ann)


# No Rebal
no_rebal = timeperiod
tkns_final, fees = fn.rebal_by_bands(timeperiod, no_rebal, prices_final, st_dollars, tgt_wghts, fee_pct)
port_val = pd.DataFrame(data=np.sum(tkns_final * prices_final, axis=1), index=prices_final.index,
                        columns=['Portfolio Values'])
port_rtns = np.divide(port_val.iloc[1:], port_val.iloc[:-1]) - 1
port_rtns.columns = ['Portfolio Rtns']

prices_last = prices_final.tail(1)
tkns_last = tkns_final.tail(1)
wghts = np.divide((tkns_last * prices_last), np.sum(tkns_last * prices_last, axis=1)[0])

port_ave, port_sd = fn.get_simple_moments_series(port_rtns, port_rtns.shape[0], port_rtns.columns[0])

x = rtns_final
y = port_rtns
model = sm.OLS(y, x)
result = model.fit()
print(result.summary())
param = result.params

pctrs2 = np.multiply(np.matmul(covar, param), param) / (port_sd ** 2)

print(pctrs)
print(pctrs2)
print(wghts)




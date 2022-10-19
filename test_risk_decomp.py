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
pctrs_imp1 = fn.pctrs(asset_sds=std_ann, asset_corrs=None, wghts=np.array(tgt_wghts), asset_covars=covar_ann)

# No Rebal Capital Weighting
no_rebal = timeperiod
tkns_final1, fees1 = fn.rebal_by_period(timeperiod, no_rebal, prices_final, st_dollars, tgt_wghts, fee_pct)
port_val1 = pd.DataFrame(data=np.sum(tkns_final1 * prices_final, axis=1), index=prices_final.index,
                         columns=['Portfolio Values Capital Weighting'])
port_rtns1 = np.divide(port_val1.iloc[1:], port_val1.iloc[:-1]) - 1
port_rtns1.columns = ['Portfolio Rtns Capital Weighting']

prices_last = prices_final.tail(1)
tkns_last1 = tkns_final1.tail(1)
wghts1 = np.divide((tkns_last1 * prices_last), np.sum(tkns_last1 * prices_last, axis=1)[0])

port_ave1, port_sd1 = fn.get_simple_moments_series(port_rtns1, port_rtns1.shape[0], port_rtns1.columns[0])

x = rtns_final
y = port_rtns1
model = sm.OLS(y, x)
result = model.fit()
# print(result.summary())
param = result.params

pctrs_regress1 = np.multiply(np.matmul(covar, param), param) / (port_sd1 ** 2)

print(tgt_wghts)
print(pctrs_imp1)
print(pctrs_regress1)
print(wghts1)

new_wghts, new_port_sd, new_pctrs = fn.calc_risk_bal_weights(asset_sds=std, asset_corrs=None, risk_tgts=tgt_wghts,
                                                             std_tgt=None, asset_covars=covar, tol=0.00001,
                                                             iter_tot=10000)

tkns_final2, fees2 = fn.rebal_by_period(timeperiod, no_rebal, prices_final, st_dollars, new_wghts, fee_pct)
port_val2 = pd.DataFrame(data=np.sum(tkns_final2 * prices_final, axis=1), index=prices_final.index,
                         columns=['Portfolio Values Capital Weighting'])
port_rtns2 = np.divide(port_val2.iloc[1:], port_val2.iloc[:-1]) - 1
port_rtns2.columns = ['Portfolio Rtns Capital Weighting']

prices_last = prices_final.tail(1)
tkns_last2 = tkns_final2.tail(1)
wghts2 = np.divide((tkns_last2 * prices_last), np.sum(tkns_last2 * prices_last, axis=1)[0])

port_ave2, port_sd2 = fn.get_simple_moments_series(port_rtns2, port_rtns2.shape[0], port_rtns2.columns[0])
x = rtns_final
y = port_rtns2
model = sm.OLS(y, x)
result = model.fit()
# print(result.summary())
param = result.params

pctrs_regress2 = np.multiply(np.matmul(covar, param), param) / (port_sd2 ** 2)

print(new_wghts)
print(new_pctrs)
print(pctrs_regress2)
print(wghts2)

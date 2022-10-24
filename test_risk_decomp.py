from api_testing import cg_pull
import functions as fn
import pandas as pd
import functools as ft
import numpy as np
import statsmodels.api as sm
import plotly.express as px

# Parameters of Risk Balancing Backtesting
token_nms = ['btc', 'eth', 'atom', 'luna']
col_nms = ['datetime'] + token_nms
start_dt = pd.Timestamp(year=2021, month=6, day=30)  # start really on the day following
stop_dt = pd.Timestamp(year=2022, month=6, day=1)  # end date really the day prior
st_dollars = 10000
fee_pct = 0.003
tgt_wghts = [0.25, 0.25, 0.25, 0.25]
rebal_freq = 90
lookback = rebal_freq
half_life = rebal_freq / 3
tgt_risk_wghts = tgt_wghts
weighting_type = 'exp'  # options are exp for exponential or arth for arithmetic
tol = 0.00001
iter_tot = 10000
int_paid = 0.02
int_rec = int_paid
needed_first_dt = start_dt - pd.Timedelta(days=(lookback + 1))

# API Pulls from Coin Gecko
btc = cg_pull('bitcoin', 'usd', 'max', 'daily')
atom = cg_pull('cosmos', 'usd', 'max', 'daily')
luna = cg_pull('terra-luna', 'usd', 'max', 'daily')
eth = cg_pull('ethereum', 'usd', 'max', 'daily')
dfs = [btc, eth, atom, luna]   # this needs to be changed if u change the tokens allocated to

# Implied Parameters
prices = ft.reduce(lambda left, right: pd.merge(left, right, on='datetime'), dfs)
prices.columns = col_nms
prices.set_index('datetime', inplace=True)
prices_full = prices.loc[np.logical_and(stop_dt >= prices.index, prices.index >= needed_first_dt)]
rtns_full = np.divide(prices_full.iloc[1:], prices_full.iloc[:-1]) - 1
prices_final = prices.loc[np.logical_and(stop_dt >= prices.index, prices.index >= start_dt)]
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

print('Target Weights of the Capital Weighted Portfolio are:')
print(tgt_wghts)
print("\n")
print('Forward Looking PCTRs of the Capital Weighted Portfolio are:')
print(pctrs_imp1)
print("\n")
print('Backward Looking PCTRs of the Capital Weighted Portfolio are:')
print(pctrs_regress1)
print("\n")
print('Final Weights of the Capital Weighted Portfolio are:')
print(wghts1)
print("\n")

new_wghts, new_port_sd, new_pctrs = fn.calc_risk_bal_weights(asset_sds=std, asset_corrs=None, risk_tgts=tgt_wghts,
                                                             std_tgt=None, asset_covars=covar, tol=0.00001,
                                                             iter_tot=10000)

tkns_final2, fees2, \
wghts_final2, cash_final2, pctrs2 = fn.rebal_by_period_risk_balancing(timeperiod=timeperiod, lookback=lookback,
                                                                      rebal_freq=rebal_freq, prices=prices_full,
                                                                      rtns=rtns_full, st_dollars=st_dollars,
                                                                      tgt_risk_wghts=tgt_risk_wghts, fee_pct=fee_pct,
                                                                      half_life=half_life, start_dt=start_dt,
                                                                      weighting_type=weighting_type, tol=tol,
                                                                      iter_tot=iter_tot, std_tgt=None,
                                                                      int_paid=int_paid, int_rec=int_rec)
port_val2 = pd.DataFrame(data=np.sum(tkns_final2 * prices_final, axis=1), index=prices_final.index,
                         columns=['Portfolio Values Risk Weighting'])
port_rtns2 = np.divide(port_val2.iloc[1:], port_val2.iloc[:-1]) - 1
port_rtns2.columns = ['Portfolio Rtns Risk Weighting']

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

print('Target Weights of the Risk Balanced Portfolio are:')
print(new_wghts)
print("\n")
print('Forward Looking PCTRs of the Risk Balanced Portfolio are:')
print(new_pctrs)
print("\n")
print('Backward Looking PCTRs of the Risk Balanced Portfolio are:')
print(pctrs_regress2)
print("\n")
print('Final Weights of the Risk Balanced Portfolio are:')
print(wghts2)
print("\n")
print('The Annualized Standard Deviation of the Tokens from ' + str(start_dt) + ' to ' + str(wghts2.index[0]))
print(std_ann)
print("\n")
print('The Correlation Matrix of the Tokens from ' + str(start_dt) + ' to ' + str(wghts2.index[0]))
print(corr)

prices.to_excel("prices.xlsx")

# Reformat for Printing and Graphing
pv_dfs = [port_val1, port_val2]
port_val_final = ft.reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), pv_dfs)
fig1 = px.line(data_frame=port_val_final)
fig1.show()
print('Final Portfolio Values Are:')
print(port_val_final.tail().to_string())

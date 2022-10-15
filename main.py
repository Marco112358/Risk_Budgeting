from api_testing import cg_pull
import functions as fn
import pandas as pd
import functools as ft
import numpy as np
import plotly.express as px
import datetime as dt

# Parameters of Risk Balancing Backtesting 
token_nms = ['btc', 'eth', 'atom', 'ada', 'sol', 'usdc']
col_nms = ['datetime'] + token_nms
start_dt = pd.Timestamp(year=2020, month=12, day=31)  # start really on the day following
st_dollars = 10000
fee_pct = 0.003
tgt_risk_wghts = [0.2, 0.35, 0.1, 0.2, 0.15]
std_tgt = 0.6
rebal_freq = 30
lookback = 90
half_life = 30
weighting_type = 'exp'  # options are exp for exponential or arth for arithmetic
tol = 0.00001
iter_tot = 10000
int_paid = 0.02
int_rec = int_paid
needed_first_dt = start_dt - pd.Timedelta(days=(lookback + 1))

# API Pulls from Coin Gecko
btc = cg_pull('bitcoin', 'usd', 'max', 'daily')
atom = cg_pull('cosmos', 'usd', 'max', 'daily')
kuji = cg_pull('kujira', 'usd', 'max', 'daily')
usdc = cg_pull('usd-coin', 'usd', 'max', 'daily')
eth = cg_pull('ethereum', 'usd', 'max', 'daily')
sol = cg_pull('solana', 'usd', 'max', 'daily')
ada = cg_pull('cardano', 'usd', 'max', 'daily')
dfs = [btc, eth, atom, ada, sol, usdc]

# Implied Parameters
prices = ft.reduce(lambda left, right: pd.merge(left, right, on='datetime'), dfs)
prices.columns = col_nms
prices.set_index('datetime', inplace=True)
prices_full = prices.loc[prices.index >= needed_first_dt]
rtns_full = np.divide(prices_full.iloc[1:], prices_full.iloc[:-1]) - 1
prices_other = prices.loc[prices.index >= start_dt]
prices_usdc = prices_other.loc[:, 'usdc']
timeperiod = prices_other.shape[0]
rtns_ex_usdc = rtns_full.drop('usdc', axis=1)
prcs_ex_usdc = prices_full.drop('usdc', axis=1)

# Run Risk Balancing BackTesting
tkns_final1, fees1, \
wghts_final1, cash_final1 = fn.rebal_by_period_risk_balancing(timeperiod=timeperiod, lookback=lookback,
                                                              rebal_freq=rebal_freq, prices=prcs_ex_usdc,
                                                              rtns=rtns_ex_usdc, st_dollars=st_dollars,
                                                              tgt_risk_wghts=tgt_risk_wghts, fee_pct=fee_pct,
                                                              half_life=half_life, start_dt=start_dt,
                                                              weighting_type=weighting_type, tol=tol,
                                                              iter_tot=iter_tot, std_tgt=std_tgt,
                                                              int_paid=int_paid, int_rec=int_rec)

# Reformat Data for Printing and Graphing
tkn_vals1 = np.sum(tkns_final1 * prices_other, axis=1)
tot_val1 = pd.concat([tkn_vals1, cash_final1], axis=1)
port_val1 = pd.DataFrame(data=np.sum(tot_val1, axis=1), index=prices_other.index,
                         columns=['Portfolio Values 30D Risk Rebal'])
fees1.columns = ['Fees 30D Risk Rebal']
port_rtns1 = np.divide(port_val1.iloc[1:], port_val1.iloc[:-1]) - 1
port_rtns1.columns = ['Portfolio Rtns 30D Risk Rebal']
total_fees1 = np.sum(fees1)

pv_dfs = [port_val1]
port_val_final = ft.reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), pv_dfs)
fig1 = px.line(data_frame=port_val_final)
fig1.show()
print(port_val_final.tail().to_string())

fees_dfs = [fees1]
fees_final = ft.reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), fees_dfs)
fees_cum = fees_final.cumsum(axis=0)
fig2 = px.line(data_frame=fees_cum)
fig2.show()
print(fees_cum.tail().to_string())

port_rtns_dfs = [port_rtns1]
port_rtns_final = ft.reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), port_rtns_dfs)

ave_rtns = pd.DataFrame(data=0, index=['annualized return', 'annualized std'], columns=port_rtns_final.columns)
for i, col in enumerate(port_rtns_final.columns):
    ave, sd = fn.get_simple_moments_series(port_rtns_final, port_rtns_final.shape[0] - 1,
                                           port_rtns_final.columns[i])
    ave_rtns.loc['annualized return', port_rtns_final.columns[i]] = ave * 365
    ave_rtns.loc['annualized std', port_rtns_final.columns[i]] = sd * np.sqrt(365)

ave_rtns.loc['ret/risk', :] = ave_rtns.loc['annualized return', :] / ave_rtns.loc['annualized std', :]
print(ave_rtns.tail().to_string())

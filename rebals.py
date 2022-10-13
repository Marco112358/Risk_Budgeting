from api_testing import cg_pull
import functions as fn
import pandas as pd
import functools as ft
import numpy as np
import plotly.express as px

token_nms = ['btc', 'eth', 'atom', 'ada', 'sol', 'usdc']
col_nms = ['datetime'] + token_nms
start_dt = '12/31/2020'  # start really on the day following
st_dollars = 10000
fee_pct = 0.003
tgt_wghts = [0.2, 0.35, 0.1, 0.2, 0.15, 0.0]
rebal_freq1 = 1
rebal_freq2 = 7
rebal_freq3 = 30
rebal_freq4 = 90
rebal_band1 = [0.05] * len(token_nms)
rebal_band2 = [0.025] * len(token_nms)
rebal_band3 = [0.1] * len(token_nms)

btc = cg_pull('bitcoin', 'usd', 'max', 'daily')
atom = cg_pull('cosmos', 'usd', 'max', 'daily')
kuji = cg_pull('kujira', 'usd', 'max', 'daily')
usdc = cg_pull('usd-coin', 'usd', 'max', 'daily')
eth = cg_pull('ethereum', 'usd', 'max', 'daily')
sol = cg_pull('solana', 'usd', 'max', 'daily')
ada = cg_pull('cardano',  'usd', 'max', 'daily')

dfs = [btc, eth, atom, ada, sol, usdc]
prices = ft.reduce(lambda left, right: pd.merge(left, right, on='datetime'), dfs)
prices.columns = col_nms
prices.set_index('datetime', inplace=True)
prices_final = prices.loc[prices.index >= start_dt]
timeperiod = prices_final.shape[0]

# Daily Rebalancing
tkns_final1, fees1 = fn.rebal_by_period(timeperiod, rebal_freq1, prices_final, st_dollars, tgt_wghts, fee_pct)
port_val1 = pd.DataFrame(data=np.sum(tkns_final1 * prices_final, axis=1), index=prices_final.index,
                         columns=['Portfolio Values Daily Rebal'])
fees1.columns = ['Fees Daily Rebal']
port_rtns1 = np.divide(port_val1.iloc[1:], port_val1.iloc[:-1]) - 1
port_rtns1.columns = ['Portfolio Rtns Daily Rebal']
total_fees1 = np.sum(fees1)

# Weekly Rebalancing
tkns_final2, fees2 = fn.rebal_by_period(timeperiod, rebal_freq2, prices_final, st_dollars, tgt_wghts, fee_pct)
port_val2 = pd.DataFrame(data=np.sum(tkns_final2 * prices_final, axis=1), index=prices_final.index,
                         columns=['Portfolio Values Weekly Rebal'])
fees2.columns = ['Fees Weekly Rebal']
port_rtns2 = np.divide(port_val2.iloc[1:], port_val2.iloc[:-1]) - 1
port_rtns2.columns = ['Portfolio Rtns Weekly Rebal']
total_fees2 = np.sum(fees2)

# 30D Rebalancing
tkns_final3, fees3 = fn.rebal_by_period(timeperiod, rebal_freq3, prices_final, st_dollars, tgt_wghts, fee_pct)
port_val3 = pd.DataFrame(data=np.sum(tkns_final3 * prices_final, axis=1), index=prices_final.index,
                         columns=['Portfolio Values 30D Rebal'])
fees3.columns = ['Fees 30D Rebal']
port_rtns3 = np.divide(port_val3.iloc[1:], port_val3.iloc[:-1]) - 1
port_rtns3.columns = ['Portfolio Rtns 30D Rebal']
total_fees3 = np.sum(fees3)

# 90D Rebalancing
tkns_final4, fees4 = fn.rebal_by_period(timeperiod, rebal_freq4, prices_final, st_dollars, tgt_wghts, fee_pct)
port_val4 = pd.DataFrame(data=np.sum(tkns_final4 * prices_final, axis=1), index=prices_final.index,
                         columns=['Portfolio Values 90D Rebal'])
fees4.columns = ['Fees 90D Rebal']
port_rtns4 = np.divide(port_val4.iloc[1:], port_val4.iloc[:-1]) - 1
port_rtns4.columns = ['Portfolio Rtns 90D Rebal']
total_fees4 = np.sum(fees4)

# 5% Tolerance Bands Rebalancing
tkns_final5, fees5 = fn.rebal_by_bands(timeperiod, rebal_band1, prices_final, st_dollars, tgt_wghts, fee_pct)
port_val5 = pd.DataFrame(data=np.sum(tkns_final5 * prices_final, axis=1), index=prices_final.index,
                         columns=['Portfolio Values 5% Rebal'])
fees5.columns = ['Fees 5% Rebal']
port_rtns5 = np.divide(port_val5.iloc[1:], port_val5.iloc[:-1]) - 1
port_rtns5.columns = ['Portfolio Rtns 5% Rebal']
total_fees5 = np.sum(fees5)

# 2.5% Tolerance Bands Rebalancing
tkns_final6, fees6 = fn.rebal_by_bands(timeperiod, rebal_band2, prices_final, st_dollars, tgt_wghts, fee_pct)
port_val6 = pd.DataFrame(data=np.sum(tkns_final6 * prices_final, axis=1), index=prices_final.index,
                         columns=['Portfolio Values 2.5% Rebal'])
fees6.columns = ['Fees 2.5% Rebal']
port_rtns6 = np.divide(port_val6.iloc[1:], port_val6.iloc[:-1]) - 1
port_rtns6.columns = ['Portfolio Rtns 2.5% Rebal']
total_fees6 = np.sum(fees6)

# 10% Tolerance Bands Rebalancing
tkns_final7, fees7 = fn.rebal_by_bands(timeperiod, rebal_band3, prices_final, st_dollars, tgt_wghts, fee_pct)
port_val7 = pd.DataFrame(data=np.sum(tkns_final7 * prices_final, axis=1), index=prices_final.index,
                         columns=['Portfolio Values 10% Rebal'])
fees7.columns = ['Fees 10% Rebal']
port_rtns7 = np.divide(port_val7.iloc[1:], port_val7.iloc[:-1]) - 1
port_rtns7.columns = ['Portfolio Rtns 10% Rebal']
total_fees7 = np.sum(fees7)

pv_dfs = [port_val1, port_val2, port_val3, port_val4, port_val5, port_val6, port_val7]
port_val_final = ft.reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), pv_dfs)
fig1 = px.line(data_frame=port_val_final)
fig1.show()
print(port_val_final.tail().to_string())

fees_dfs = [fees1, fees2, fees3, fees4, fees5, fees6, fees7]
fees_final = ft.reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), fees_dfs)
fees_cum = fees_final.cumsum(axis=0)
fig2 = px.line(data_frame=fees_cum)
fig2.show()
print(fees_cum.tail().to_string())

port_rtns_dfs = [port_rtns1, port_rtns2, port_rtns3, port_rtns4, port_rtns5, port_rtns6, port_rtns7]
port_rtns_final = ft.reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), port_rtns_dfs)

ave_rtns = pd.DataFrame(data=0, index=['annualized return', 'annualized std'], columns=port_rtns_final.columns)
for i, col in enumerate(port_rtns_final.columns):
    ave, sd = fn.get_simple_moments(port_rtns_final, port_rtns_final.shape[0] - 1,
                                    port_rtns_final.columns[i])
    ave_rtns.loc['annualized return', port_rtns_final.columns[i]] = ave * 365
    ave_rtns.loc['annualized std', port_rtns_final.columns[i]] = sd * np.sqrt(365)

ave_rtns.loc['ret/risk', :] = ave_rtns.loc['annualized return', :] / ave_rtns.loc['annualized std', :]
print(ave_rtns.tail().to_string())

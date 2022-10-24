from api_testing import cg_pull
import functions as fn
import pandas as pd
import functools as ft
import numpy as np
import plotly.express as px
import datetime
import matplotlib.pyplot as plt
import openpyxl

# Parameters of Risk Balancing Backtesting
token_nms = ['btc', 'eth', 'atom', 'ada', 'sol']
col_nms = ['datetime'] + token_nms
start_dt = pd.Timestamp(year=2020, month=12, day=31)  # start really on the day following
st_dollars = 10000
fee_pct = 0.003
tgt_risk_wghts = [0.2, 0.35, 0.1, 0.2, 0.15]
std_tgt = None
rebal_freq = 90
lookback = rebal_freq
half_life = rebal_freq / 3
weighting_type = 'exp'  # options are exp for exponential or arth for arithmetic
tol = 0.00001
iter_tot = 10000
int_paid = 0.02
int_rec = int_paid
needed_first_dt = start_dt - pd.Timedelta(days=(lookback + 1))

if std_tgt is not None:
    std_tgt_unannualized = std_tgt / np.sqrt(365)
else:
    std_tgt_unannualized = None

# API Pulls from Coin Gecko
btc = cg_pull('bitcoin', 'usd', 'max', 'daily')
atom = cg_pull('cosmos', 'usd', 'max', 'daily')
kuji = cg_pull('kujira', 'usd', 'max', 'daily')
usdc = cg_pull('usd-coin', 'usd', 'max', 'daily')
eth = cg_pull('ethereum', 'usd', 'max', 'daily')
sol = cg_pull('solana', 'usd', 'max', 'daily')
ada = cg_pull('cardano', 'usd', 'max', 'daily')
dfs = [btc, eth, atom, ada, sol]  # this needs to be changed if u change the tokens allocated to

# Implied Parameters
prices = ft.reduce(lambda left, right: pd.merge(left, right, on='datetime'), dfs)
prices.columns = col_nms
prices.set_index('datetime', inplace=True)
prices_full = prices.loc[prices.index >= needed_first_dt]
rtns_full = np.divide(prices_full.iloc[1:], prices_full.iloc[:-1]) - 1
prices_final = prices.loc[prices.index >= start_dt]
rtns_final = np.divide(prices_final.iloc[1:], prices_final.iloc[:-1]) - 1

# prices_usdc = prices_other.loc[:, 'usdc']
timeperiod = prices_final.shape[0]
rtn_timeperiod = rtns_final.shape[0]
# rtns_ex_usdc = rtns_full.drop('usdc', axis=1)
# prcs_ex_usdc = prices_full.drop('usdc', axis=1)

# Run Risk Balancing BackTesting
tkns_final1, fees1, \
wghts_final1, cash_final1, pctrs1 = fn.rebal_by_period_risk_balancing(timeperiod=timeperiod, lookback=lookback,
                                                                      rebal_freq=rebal_freq, prices=prices_full,
                                                                      rtns=rtns_full, st_dollars=st_dollars,
                                                                      tgt_risk_wghts=tgt_risk_wghts, fee_pct=fee_pct,
                                                                      half_life=half_life, start_dt=start_dt,
                                                                      weighting_type=weighting_type, tol=tol,
                                                                      iter_tot=iter_tot, std_tgt=std_tgt_unannualized,
                                                                      int_paid=int_paid, int_rec=int_rec)

# Reformat Data for Printing and Graphing
tkn_vals1 = np.sum(tkns_final1 * prices_final, axis=1)
tot_val1 = pd.concat([tkn_vals1, cash_final1], axis=1)
port_val1 = pd.DataFrame(data=np.sum(tot_val1, axis=1), index=prices_final.index,
                         columns=['Portfolio Values 90D Risk Rebal'])
fees1.columns = ['Fees 90D Risk Rebal']
port_rtns1 = np.divide(port_val1.iloc[1:], port_val1.iloc[:-1]) - 1
port_rtns1.columns = ['Portfolio Rtns 90D Risk Rebal']
total_fees1 = np.sum(fees1)

# Parameters
tgt_wghts = tgt_risk_wghts
rebal_freq1 = 1
rebal_freq2 = 7
rebal_freq3 = 30
rebal_freq4 = 90
rebal_band1 = [0.05] * len(token_nms)
rebal_band2 = [0.025] * len(token_nms)
rebal_band3 = [0.1] * len(token_nms)
relative_rebal_band1 = 0.1
relative_rebal_band2 = 0.25
no_rebal = timeperiod

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

# 10% Relative Tolerance Bands Rebalancing
tkns_final8, fees8 = fn.rebal_by_bands(timeperiod, None, prices_final, st_dollars, tgt_wghts, fee_pct,
                                       relative_rebal_band1)
port_val8 = pd.DataFrame(data=np.sum(tkns_final8 * prices_final, axis=1), index=prices_final.index,
                         columns=['Portfolio Values Relative 10% Rebal'])
fees8.columns = ['Fees Relative 10% Rebal']
port_rtns8 = np.divide(port_val8.iloc[1:], port_val8.iloc[:-1]) - 1
port_rtns8.columns = ['Portfolio Rtns Relative 10% Rebal']
total_fees8 = np.sum(fees8)

# 25% Relative Tolerance Bands Rebalancing
tkns_final9, fees9 = fn.rebal_by_bands(timeperiod, None, prices_final, st_dollars, tgt_wghts, fee_pct,
                                       relative_rebal_band2)
port_val9 = pd.DataFrame(data=np.sum(tkns_final9 * prices_final, axis=1), index=prices_final.index,
                         columns=['Portfolio Values Relative 25% Rebal'])
fees9.columns = ['Fees Relative 25% Rebal']
port_rtns9 = np.divide(port_val9.iloc[1:], port_val9.iloc[:-1]) - 1
port_rtns9.columns = ['Portfolio Rtns Relative 25% Rebal']
total_fees9 = np.sum(fees9)

# No Rebal
tkns_final10, fees10 = fn.rebal_by_bands(timeperiod, no_rebal, prices_final, st_dollars, tgt_wghts, fee_pct)
port_val10 = pd.DataFrame(data=np.sum(tkns_final10 * prices_final, axis=1), index=prices_final.index,
                         columns=['Portfolio Values No Rebalance'])
fees10.columns = ['Fees No Rebalance']
port_rtns10 = np.divide(port_val10.iloc[1:], port_val10.iloc[:-1]) - 1
port_rtns10.columns = ['Portfolio Rtns No Rebalance']
total_fees10 = np.sum(fees10)

# Reformat for Printing and Graphing
pv_dfs = [port_val1, port_val2, port_val3, port_val4, port_val5, port_val6, port_val7, port_val8, port_val9, port_val10]
port_val_final = ft.reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), pv_dfs)
fig1 = px.line(data_frame=port_val_final)
fig1.show()
print('Final Portfolio Values Are:')
print(port_val_final.tail().to_string())

fees_dfs = [fees1, fees2, fees3, fees4, fees5, fees6, fees7, fees8, fees9, fees10]
fees_final = ft.reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), fees_dfs)
fees_cum = fees_final.cumsum(axis=0)
fig2 = px.line(data_frame=fees_cum)
fig2.show()
print('Final Cumulative Fees Paid Are:')
print(fees_cum.tail().to_string())

port_rtns_dfs = [port_rtns1, port_rtns2, port_rtns3, port_rtns4, port_rtns5, port_rtns6, port_rtns7, port_rtns8,
                 port_rtns9, port_rtns10]
port_rtns_final = ft.reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), port_rtns_dfs)

ave_rtns = pd.DataFrame(data=0, index=['annualized return', 'annualized std'], columns=port_rtns_final.columns)
for i, col in enumerate(port_rtns_final.columns):
    ave, sd = fn.get_simple_moments_series(port_rtns_final, port_rtns_final.shape[0], port_rtns_final.columns[i])
    ave_rtns.loc['annualized return', port_rtns_final.columns[i]] = ave * 365
    ave_rtns.loc['annualized std', port_rtns_final.columns[i]] = sd * np.sqrt(365)

print('Average Returns and Risks Are:')
ave_rtns.loc['ret/risk', :] = ave_rtns.loc['annualized return', :] / ave_rtns.loc['annualized std', :]
print(ave_rtns.tail().to_string())

HPR = np.divide(port_val_final, st_dollars) - 1
print('Final Holding Period Returns Are:')
print(HPR.tail(1).to_string())

prices.to_excel("prices.xlsx")

# Compare 30D Rebal to No Rebalance on 500 Different Start Dates
n_days = 500
timestamp_list = [start_dt + datetime.timedelta(days=x) for x in range(n_days)]
date_list = [x.date() for x in timestamp_list]
diff_final = pd.DataFrame(index=prices_final.index, columns=timestamp_list)

for i, dt in enumerate(timestamp_list):
    prices_trim = prices_full.loc[prices_full.index >= dt]
    timeperiod = prices_trim.shape[0]
    # 30D Rebal
    tkns_final3, fees3 = fn.rebal_by_period(timeperiod, rebal_freq3, prices_trim, st_dollars, tgt_wghts, fee_pct)
    port_val3 = pd.DataFrame(data=np.sum(tkns_final3 * prices_trim, axis=1), index=prices_trim.index,
                             columns=['Portfolio Values 30D Rebal'])
    # No Rebal
    tkns_final10, fees10 = fn.rebal_by_period(timeperiod, no_rebal, prices_trim, st_dollars, tgt_wghts, fee_pct)
    port_val10 = pd.DataFrame(data=np.sum(tkns_final10 * prices_trim, axis=1), index=prices_trim.index,
                              columns=['Portfolio Values No Rebalance'])
    # Create the dataset of Rebal / No Rebal (value > 1 means Rebal better on that date
    mrg = pd.concat([port_val3, port_val10], axis=1)
    diff = mrg.iloc[:, 0] / mrg.iloc[:, 1]
    diff_final[dt] = diff
diff_final.fillna(value=1, inplace=True)

diff_final.columns = date_list

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()
ax.plot(diff_final.index, diff_final.loc[:, date_list], color='b', linewidth=0.05)
ax.grid()
fig.tight_layout()
plt.show()

last_diff = diff_final.tail(1)
pct_above_one = last_diff[last_diff > 1].count(axis=1)[0] / last_diff.shape[1]
print('Percentage of Rebalanced Portfolios that end Greater Than No Rebalanced Portfolios starting every day from '
      + str(start_dt.date()) + ' to ' + str(date_list[len(date_list) - 1]) + ' is ' + str(pct_above_one * 100) + '%')

### START TESTING SIMULATED RETURN SERIES ###
ave, std, covar, corr = fn.get_simple_moments(rtns_final, rtn_timeperiod)
data = pd.DataFrame(data=np.random.multivariate_normal(ave, covar, size=100), columns=ave.index)
(1+data).product()


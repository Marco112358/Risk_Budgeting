from api_testing import cg_pull
import pandas as pd
import functools as ft
import numpy as np
import functions as fn
import matplotlib.pyplot as plt


token_nms = ['btc', 'eth', 'atom']
col_nms = ['datetime'] + token_nms
st_dollars = 10000
fee_pct = 0.003
tgt_wghts = [0.33333333, 0.33333333, 0.33333333]
st_price = 100
sample_len = 30
no_samples = 12
no_trials = 100
prc_st_list = [st_price] * len(token_nms)

# API Pulls from Coin Gecko
btc = cg_pull('bitcoin', 'usd', 'max', 'daily')
atom = cg_pull('cosmos', 'usd', 'max', 'daily')
# kuji = cg_pull('kujira', 'usd', 'max', 'daily')
# usdc = cg_pull('usd-coin', 'usd', 'max', 'daily')
eth = cg_pull('ethereum', 'usd', 'max', 'daily')
# sol = cg_pull('solana', 'usd', 'max', 'daily')
# ada = cg_pull('cardano', 'usd', 'max', 'daily')
dfs = [btc, eth, atom]  # this needs to be changed if u change the tokens allocated to

# Implied Parameters
timeperiod = sample_len * no_samples
prices = ft.reduce(lambda left, right: pd.merge(left, right, on='datetime'), dfs)
prices.columns = col_nms
prices.set_index('datetime', inplace=True)
rtns = np.divide(prices.iloc[1:], prices.iloc[:-1]) - 1

# Historical Sample
hist_prc_samp, hist_rtn_samp = fn.hist_sample(rtns, sample_len, no_samples, prc_st_list)


# Compare 30D Rebal to No Rebalance on 500 Different Start Dates
rebal_freq3 = 30
no_rebal = timeperiod
n_days = sample_len * no_samples
index_list = np.arange(0, n_days)
diff_final = pd.DataFrame(index=hist_prc_samp.index, columns=index_list)

for i in index_list:
    prices_trim = hist_prc_samp.loc[hist_prc_samp.index >= i]
    fwd_time = prices_trim.shape[0]
    # 30D Rebal
    tkns_final3, fees3 = fn.rebal_by_period(fwd_time, rebal_freq3, prices_trim, st_dollars, tgt_wghts, fee_pct)
    port_val3 = pd.DataFrame(data=np.sum(tkns_final3 * prices_trim, axis=1), index=prices_trim.index,
                             columns=['Portfolio Values 30D Rebal'])
    # No Rebal
    tkns_final10, fees10 = fn.rebal_by_period(fwd_time, no_rebal, prices_trim, st_dollars, tgt_wghts, fee_pct)
    port_val10 = pd.DataFrame(data=np.sum(tkns_final10 * prices_trim, axis=1), index=prices_trim.index,
                              columns=['Portfolio Values No Rebalance'])
    # Create the dataset of Rebal / No Rebal (value > 1 means Rebal better on that date
    mrg = pd.concat([port_val3, port_val10], axis=1)
    diff = mrg.iloc[:, 0] / mrg.iloc[:, 1]
    diff_final[i] = diff
diff_final.fillna(value=1, inplace=True)

diff_final.columns = index_list

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()
ax.plot(diff_final.index, diff_final.loc[:, index_list], color='b', linewidth=0.05)
ax.grid()
fig.tight_layout()
plt.show()

last_diff = diff_final.tail(1)
pct_above_one = last_diff[last_diff > 1].count(axis=1) / last_diff.shape[1]
print('Percentage of Rebalanced Portfolios that end Greater Than No Rebalanced Portfolios starting every day is '
      + str(pct_above_one * 100) + '%')


# Loop Through No Trials
rebal_freq3 = 30
port_nms = ['Portfolio Values 30D Rebal', 'Portfolio Values No Rebalance']
no_rebal = timeperiod
diff_final = pd.DataFrame(index=np.arange(0, timeperiod), columns=np.arange(0, no_trials))
pv_finals = pd.DataFrame(index=np.arange(0, no_trials), columns=port_nms)
for i in np.arange(0, no_trials):
    hist_prc_samp, hist_rtn_samp = fn.hist_sample(rtns, sample_len, no_samples, prc_st_list)
    # 30D Rebal
    tkns_final3, fees3 = fn.rebal_by_period(timeperiod, rebal_freq3, hist_prc_samp, st_dollars, tgt_wghts, fee_pct)
    port_val3 = pd.DataFrame(data=np.sum(tkns_final3 * hist_prc_samp, axis=1), index=hist_prc_samp.index,
                             columns=[port_nms[0]])
    # No Rebal
    tkns_final10, fees10 = fn.rebal_by_period(timeperiod, no_rebal, hist_prc_samp, st_dollars, tgt_wghts, fee_pct)
    port_val10 = pd.DataFrame(data=np.sum(tkns_final10 * hist_prc_samp, axis=1), index=hist_prc_samp.index,
                              columns=[port_nms[1]])
    # Create the dataset of Rebal / No Rebal (value > 1 means Rebal better on that date
    mrg = pd.concat([port_val3, port_val10], axis=1)
    diff = mrg.iloc[:, 0] / mrg.iloc[:, 1]
    diff_final[i] = diff
    pv_finals.iloc[i, :] = mrg.tail(1)

# Add Absolute Difference between Rebalanced and No Rebalance
pv_finals.loc[:, 'Difference'] = pv_finals.iloc[:, 0] - pv_finals.iloc[:, 1]
port_nms.append('Difference')

# Plot the Relative difference (rebal / no rebal) across all trials for each day
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()
ax.plot(diff_final.index, diff_final.loc[:, np.arange(0, no_trials)], color='b', linewidth=0.05)
ax.grid()
fig.tight_layout()
plt.show()

# Plot the Final Value of Each Portfolio (rebal and no rebal), and the Absolute Difference across all Trials
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()
ax.plot(pv_finals.index, pv_finals, label=port_nms)
plt.legend(loc='lower left')
ax.grid()
fig.tight_layout()
plt.show()

# Print the % of Rebal Portfolios that end > no Rebal Portfolios
last_diff = diff_final.tail(1)
pct_above_one = last_diff[last_diff > 1].count(axis=1) / last_diff.shape[1]
print('Percentage of Rebalanced Portfolios that end Greater Than No Rebalanced Portfolios starting every day is '
      + str(pct_above_one * 100) + '%')

# Print the Average and Median Absolute Difference Between Rebal and No Rebal Portfolios
ave_diff = np.average(pv_finals.loc[:, 'Difference'])
med_diff = np.median(pv_finals.loc[:, 'Difference'])
print('Average Difference between Rebalanced Portfolios versus No Rebalanced Portfolios: $'
      + str(round(ave_diff)))
print('Median Difference between Rebalanced Portfolios versus No Rebalanced Portfolios: $'
      + str(round(med_diff)))

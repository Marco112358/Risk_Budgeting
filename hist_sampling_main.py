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



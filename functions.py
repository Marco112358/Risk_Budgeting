import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_exponential_moments(df=None, half_life=0.0, lookback=100):
    # parameters: arr = array of data to get ema and exp_std;
    # half_life is the half life of the exponential you want to use, period is in the period of the time series
    # for example: if you put in 100 days of data, and you want half of the weight to come from the 1st 20 days, put 20
    data = df.tail(lookback)
    cols = data.columns
    ewm = data.ewm(halflife=half_life, min_periods=lookback)
    ema = ewm.mean().tail(1)
    exp_std = ewm.std().tail(1)

    # get the exponentially weighted covariance matrix
    exp_covar = ewm.cov().unstack(level=1).tail(1)
    cov_cols_all = exp_covar.columns
    cov_cols = []
    cov_rows = []
    for i in np.arange(0, len(cov_cols_all)):
        if cov_cols_all[i][0] not in cov_cols:
            cov_cols.append(cov_cols_all[i][0])
        if cov_cols_all[i][1] not in cov_rows:
            cov_rows.append(cov_cols_all[i][1])
    cov_mat = np.matrix(exp_covar).reshape(len(cols), len(cols)).transpose()
    cov_out = pd.DataFrame(data=cov_mat, index=cov_rows, columns=cov_cols)
    cov_out = cov_out.reindex(cov_cols)

    # get the exponentially weighted correlation matrix
    exp_corr = ewm.corr().unstack(level=1).tail(1)
    corr_cols_all = exp_corr.columns
    corr_cols = []
    corr_rows = []
    for i in np.arange(0, len(corr_cols_all)):
        if corr_cols_all[i][0] not in corr_cols:
            corr_cols.append(corr_cols_all[i][0])
        if corr_cols_all[i][1] not in corr_rows:
            corr_rows.append(corr_cols_all[i][1])
    corr_mat = np.matrix(exp_corr).reshape(len(cols), len(cols)).transpose()
    corr_out = pd.DataFrame(data=corr_mat, index=corr_rows, columns=corr_cols)
    corr_out = corr_out.reindex(corr_cols)

    return ema, exp_std, cov_out, corr_out


def get_simple_moments(df=None, lookback=100):
    data = df.tail(lookback)
    ave = data.mean()
    std = data.std()
    covar = data.cov()
    corr = data.corr()
    return ave, std, covar, corr


def get_simple_moments_series(df=None, lookback=100,  col_nm='na'):
    data = df.tail(lookback)
    arry = np.array(data.loc[:, col_nm])
    prc_ave = np.average(arry)
    prc_std = np.std(arry)
    return prc_ave, prc_std


def annualize_moments(ave=None, std=None, covar=None, period=1):
    ave_out = ave * (365 / period)
    std_out = std * np.sqrt((365 / period))
    cov_out = covar * (365 / period)
    return ave_out, std_out, cov_out


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


def rebal_by_bands(timeperiod=100, rebal_bands=None, prices_final=None, st_dollars=10000, tgt_wghts=None, fee_pct=0.0,
                   relative_rebal_band=None):
    st_vals = np.multiply(st_dollars, tgt_wghts)
    st_prcs = prices_final.head(n=1)
    st_tkns = st_vals / st_prcs
    tkns_final = pd.DataFrame(index=prices_final.index, columns=prices_final.columns)
    fees = pd.DataFrame(data=0, index=prices_final.index, columns=['Fees'])
    tkns_final.iloc[0] = st_tkns
    if rebal_bands is None and relative_rebal_band is not None:
        mins = np.subtract(tgt_wghts, np.multiply(relative_rebal_band, tgt_wghts))
        maxes = np.add(tgt_wghts, np.multiply(relative_rebal_band, tgt_wghts))
    elif rebal_bands is not None and relative_rebal_band is None:
        mins = np.subtract(tgt_wghts, rebal_bands)
        mins = mins.clip(min=0.0)
        maxes = np.add(tgt_wghts, rebal_bands)
        maxes = maxes.clip(max=1.0)
    else:
        print('conflicting rebalance bands inputs')
        return
    for i in np.arange(1, timeperiod):
        prcs = prices_final.iloc[i]
        act_vals = tkns_final.iloc[i - 1] * prcs
        pv_prefee = np.sum(act_vals)
        wghts = act_vals / pv_prefee
        if any(wghts < mins) or any(wghts > maxes):
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


# Newton Algo
def newtons_algo(f, df, x0, tol, iter_tot):
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


def calc_risk_bal_weights(asset_sds=None, asset_corrs=None, risk_tgts=None, std_tgt=None, asset_covars=None,
                          tol=0.00001, iter_tot=10000):
    if asset_covars is None:
        asset_covars = np.multiply(asset_sds.transpose(), np.multiply(asset_corrs, asset_sds))

    # Functions used in newton algo
    fn = lambda x: np.matmul(asset_covars, x.transpose()) - (risk_tgts / x).transpose()
    f_prime = lambda x: asset_covars + np.diagflat((risk_tgts / np.multiply(x, x)))

    # Initial Guess for Algo
    x0_test = np.zeros(len(asset_sds)) + (1/len(asset_sds))

    # Run Algo, get weights
    x_out, itn = newtons_algo(fn, f_prime, x0_test, tol, iter_tot)
    wghts_out = x_out / np.matmul(np.matrix(np.ones(x_out.shape[0])), x_out.transpose())
    port_sd = np.sqrt(np.matmul(wghts_out, np.matmul(asset_covars, wghts_out.transpose())))[0][0]
    mctrs = np.matmul(asset_covars, wghts_out.transpose()) / port_sd
    ctrs = np.multiply(mctrs, wghts_out.transpose())
    pctrs = ctrs / port_sd

    # lever or unlever the portfolio depending on the desired target standard deviation
    if std_tgt is not None:
        # Levering the Portfolio Weights to get to the targeted total portfolio standard deviation level
        lev = std_tgt / port_sd
        new_wghts = lev * wghts_out
        new_port_sd = np.sqrt(np.matmul(new_wghts, np.matmul(asset_covars, new_wghts.transpose())))[0][0]
        new_mctrs = np.matmul(asset_covars, new_wghts.transpose()) / new_port_sd
        new_ctrs = np.multiply(new_mctrs, new_wghts.transpose())
        new_pctrs = new_ctrs / new_port_sd
    else:
        new_wghts = wghts_out
        new_port_sd = port_sd
        new_pctrs = pctrs

    new_wghts = pd.Series(data=np.array(new_wghts.transpose()).reshape(len(asset_sds)), index=asset_sds.index)
    return new_wghts, new_port_sd, new_pctrs


def rebal_by_period_risk_balancing(timeperiod=100, lookback=90, rebal_freq=None, prices=None, rtns=None, st_dollars=10000,
                                   tgt_risk_wghts=None, fee_pct=0.0, half_life=30, start_dt=None,
                                   weighting_type='arth', tol=0.00001, iter_tot=10000, std_tgt=None, int_paid=0.0,
                                   int_rec=0.0):

    rtn_df = rtns.loc[rtns.index <= start_dt]
    if weighting_type == 'arth':
        ave, std, covar, corr = get_simple_moments(df=rtn_df, lookback=lookback)
    elif weighting_type == 'exp':
        ave, std, covar, corr = get_exponential_moments(df=rtn_df, half_life=half_life, lookback=lookback)
        ave = ave.iloc[0]
        std = std.iloc[0]
    st_ave, st_std, st_cov = annualize_moments(ave=ave, std=std, covar=covar, period=1)
    new_wghts, new_port_sd, new_pctrs = calc_risk_bal_weights(asset_sds=st_std, asset_corrs=None,
                                                              risk_tgts=tgt_risk_wghts, std_tgt=std_tgt,
                                                              asset_covars=st_cov, tol=tol, iter_tot=iter_tot)

    st_vals = np.multiply(st_dollars, new_wghts)
    cash_prev = st_dollars - np.sum(st_vals)
    prices = prices.loc[prices.index >= start_dt]
    st_prcs = prices.head(n=1)
    st_tkns = st_vals / st_prcs
    tkns_final = pd.DataFrame(index=prices.index, columns=prices.columns)
    wghts_final = pd.DataFrame(index=prices.index, columns=prices.columns)
    cash_final = pd.DataFrame(index=prices.index, columns=['Cash'])
    fees = pd.DataFrame(data=0, index=prices.index, columns=['Fees'])
    tkns_final.iloc[0] = st_tkns
    wghts_final.iloc[0] = new_wghts
    cash_final.iloc[0] = cash_prev
    rebals = np.arange(rebal_freq, timeperiod, step=rebal_freq)
    for i in np.arange(1, timeperiod):
        if i in rebals:
            rtn_df = rtns.loc[rtns.index <= prices.index[i]]
            if weighting_type == 'arth':
                ave, std, covar, corr = get_simple_moments(df=rtn_df, lookback=lookback)
            elif weighting_type == 'exp':
                ave, std, covar, corr = get_exponential_moments(df=rtn_df, half_life=half_life, lookback=lookback)
                ave = ave.iloc[0]
                std = std.iloc[0]
            st_ave, st_std, st_cov = annualize_moments(ave=ave, std=std, covar=covar, period=1)
            new_wghts, new_port_sd, new_pctrs = calc_risk_bal_weights(asset_sds=st_std, asset_corrs=None,
                                                                      risk_tgts=tgt_risk_wghts, std_tgt=std_tgt,
                                                                      asset_covars=st_cov, tol=tol, iter_tot=iter_tot)
            prcs = prices.iloc[i]
            act_vals = tkns_final.iloc[i - 1] * prcs
            pv_prefee = np.sum(act_vals) + cash_final.iloc[i - 1][0]  # correct value with cash
            tgt_vals_prefee = np.multiply(pv_prefee, new_wghts)  # below pv_prefee
            diffs = act_vals - tgt_vals_prefee
            total_trade_val = np.sum(np.abs(diffs))
            fee = total_trade_val * fee_pct
            fees.iloc[i] = fee
            pv_postfee = pv_prefee - fee
            tgt_vals_postfee = np.multiply(pv_postfee, new_wghts)
            tkns_final.iloc[i] = tgt_vals_postfee / prcs
            wghts_final.iloc[i] = new_wghts
            cash_final.iloc[i] = pv_postfee - np.sum(tgt_vals_postfee)
        else:
            tkns_final.iloc[i] = tkns_final.iloc[i - 1]
            wghts_final.iloc[i] = wghts_final.iloc[i - 1]
            if cash_final.iloc[i - 1][0] < 0:
                cash_final.iloc[i] = cash_final.iloc[i - 1][0] * (1 + int_paid / 365)
            else:
                cash_final.iloc[i] = cash_final.iloc[i - 1][0] * (1 + int_rec / 365)
    return tkns_final, fees, wghts_final, cash_final

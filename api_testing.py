
from pycoingecko import CoinGeckoAPI
import pandas as pd
from datetime import datetime
from datetime import timezone

cg = CoinGeckoAPI()

def cg_pull(ticker='btc', curr='usd', days='max', intv='daily'):
    out = cg.get_coin_market_chart_by_id(id=ticker, vs_currency=curr, days=days, interval=intv)
    df = pd.DataFrame(data=out)
    df[['date', 'price']] = pd.DataFrame(df.prices.tolist(), index=df.index)
    df[['date']] = df[['date']] / 1000
    df['datetime'] = [datetime.fromtimestamp(x) for x in df['date']]
    df_out = df[['datetime', 'price']]
    return df_out

btc = cg_pull('bitcoin', 'usd', 'max', 'daily')
atom = cg_pull('cosmos', 'usd', 'max', 'daily')
kuji = cg_pull('kujira', 'usd', 'max', 'daily')
usdc = cg_pull('usd-coin', 'usd', 'max', 'daily')
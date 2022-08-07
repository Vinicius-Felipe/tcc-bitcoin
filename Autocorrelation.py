import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def autocorrelation(series, start ='2010-01-01', end = '2022-12-31'):
    series_period = series.loc[pd.to_datetime(start):pd.to_datetime(end)]
    autocorr_plot1,ax1=plt.subplots(figsize=(8,7))
    ax1.set_xlabel('Lag')
    autocorr_plot1=plot_acf(series_period,ax=ax1, label='BTC Autocorrelation', lags = len(series_period) - 1)
    ax1.set_title('Autocorrelação de ' + start + ' até ' + end)
    plt.show()
    return

def plot_autocorrelations():
    
    autocorrelation(btc_price)

    autocorrelation(btc_price, '2012-11-28', '2020-05-11')

    autocorrelation(btc_price, '2012-11-28', '2016-07-09')

    autocorrelation(btc_price, '2016-07-09', '2020-05-11')

    autocorrelation(btc_price, '2020-05-11')

    days_in_actual_halving = btc_price.index[-1] - pd.to_datetime('2020-05-11')
    autocorrelation(btc_price, str(pd.to_datetime('2016-07-09') + days_in_actual_halving)[:10])
    
    return

btc = pd.read_csv('btc.csv')
btc = btc.rename(columns={"time": "date"})
btc = btc.set_index('date')
btc_price = pd.DataFrame(btc['PriceUSD'], index = btc.index)
btc_price = btc_price.dropna()
btc_price.index = pd.to_datetime(btc_price.index)


if __name__ == '__main__':
  plot_autocorrelations()
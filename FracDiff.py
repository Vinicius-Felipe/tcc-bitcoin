import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
pd.options.plotting.backend = "plotly"

def period(series, year):
  series_period = series[series.index.year >= year]
  return series_period

# Code from the book 'Advances in Financial Machine Learning
def getWeights(d,size):
  # thres > 0 drops insignificant weights
  w=[1.]
  for k in range(1,size):
    w_ = -w[-1]/k*(d-k+1)
    w.append(w_)
  w=np.array(w[::-1]).reshape(-1,1)
  return w

# Code from the book 'Advances in Financial Machine Learning
def fracDiff (series, d, thres=.01):
  '''
  Increasing width window, with treatment of Nas
  Note 1: For thres=1, nothing is skipped.
  Note 2: d can be any positive fractional, not necessarily bounded [0,1].
  '''
  #1) Compute weights for the longest series
  w=getWeights (d, series.shape[0])
  #2) Determine initial calcs to be skipped based on weight-loss threshold
  w_ = np.cumsum(abs(w))
  w_/=w_[-1]
  skip = w_[w_>thres].shape[0]
  #3) Apply weights to values
  df = {}
  for name in series.columns:
    seriesF, df_= series[[name]].fillna(method='ffill').dropna(), pd.Series()
    for iloc in range (skip, seriesF.shape[0]):
      loc=seriesF.index[iloc]
      if not np.isfinite(series.loc[loc, name]) :continue #exclude NAs
      df_[loc] = np.dot(w[-(iloc + 1):,:].T, seriesF.loc[:loc])[0,0]
    df[name] =df_.copy(deep=True)
  df = pd.concat(df,axis=1)
  return df

def adf_curve(ser):
  range_idx = int(100)
  pearson = []
  adf = []
  spearman = []
  d_opt = None
  for d in range(range_idx):
    frac = fracDiff(ser, (d+1)/range_idx, 0.05)
    pearson.append(frac.corrwith(ser)[0])
    adf_t = adfuller(frac.iloc[:,0], maxlag = 1, regression = 'ct', autolag = None)[0]
    adf.append(adf_t)
    spearman.append(frac.corrwith(ser, method = 'spearman')[0])
    if d_opt is None:
      if adf_t < adf_95:
        d_opt = (d + 1)/100
        pearson_opt = frac.corrwith(ser)[0]
        spearman_opt = frac.corrwith(ser, method = 'spearman')[0]
  from plotly.subplots import make_subplots
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(go.Scatter(x=np.linspace(1,range_idx,range_idx)/range_idx , y=adf , name="ADF",), secondary_y=False,)
  fig.add_trace(go.Scatter(x=np.linspace(1,range_idx,range_idx)/range_idx , y=pearson , name="Pearson",), secondary_y=True,)
  fig.add_trace(go.Scatter(x=np.linspace(1,range_idx,range_idx)/range_idx , y=spearman , name="Spearman",), secondary_y=True,)
  fig.add_hline(y=adf_95)
  fig.show()
  return d_opt, pearson_opt, spearman_opt


btc = pd.read_csv('btc.csv')
btc = btc.rename(columns={"time": "date"})
btc = btc.set_index('date')
btc_price = pd.DataFrame(btc['PriceUSD'], index = btc.index)
btc_price = btc_price.dropna()
btc_price.index = pd.to_datetime(btc_price.index)
btc_price_2017 = period(btc_price, 2017)
btc_price_2020 = period(btc_price, 2020)

nasdaq = pd.read_csv('nasdaq-100.csv', on_bad_lines='skip')
nasdaq = nasdaq.set_index('date')
nasdaq.index = pd.to_datetime(nasdaq.index)
nasdaq_2017 = period(nasdaq, 2017)
nasdaq_2020 = period(nasdaq, 2020)

gold = pd.read_csv('gold.csv')
gold = gold.rename(columns={"Date": "date"})
gold = gold.set_index('date')
gold.index = pd.to_datetime(gold.index)
gold = gold['Price'].str.replace(',', '').astype(float)
gold = gold.to_frame()
gold_2017 = period(gold, 2017)
gold_2020 = period(gold, 2020)

frac = fracDiff(btc_price, 0.4, 0.05)
adf = adfuller(frac.iloc[:,0], regression='ct')
adf_95 = adf[4]['5%']

def print_correlation():
  bit = adf_curve(btc_price)
  nas = adf_curve(nasdaq)
  gol = adf_curve(gold)

  print('BTC', bit)
  print('NASDAQ', nas)
  print('Ouro',gol)

  btc_frac = fracDiff(btc_price, bit[0], 0.05)
  nasdaq_frac = fracDiff(nasdaq, nas[0], 0.05)
  gold_frac = fracDiff(gold, gol[0], 0.05)
  print('Correlacao de Pearson BTC x NASDAQ-100 período inteiro:', btc_frac.corrwith(nasdaq_frac[nasdaq_frac.columns[0]]))
  print('Correlacao de Pearson BTC x ouro período inteiro:', btc_frac.corrwith(gold_frac[gold_frac.columns[0]]))
  print('Correlacao de Spearman BTC x NASDAQ-100 período inteiro:', btc_frac.corrwith(nasdaq_frac[nasdaq_frac.columns[0]], method = 'spearman'))
  print('Correlacao de Spearman BTC x ouro período inteiro:', btc_frac.corrwith(gold_frac[gold_frac.columns[0]], method = 'spearman'))

  print('')

  bit = adf_curve(btc_price_2017)
  nas = adf_curve(nasdaq_2017)
  gol = adf_curve(gold_2017)

  print('BTC', bit)
  print('NASDAQ', nas)
  print('Ouro',gol)

  btc_frac = fracDiff(btc_price_2017, bit[0], 0.05)
  nasdaq_frac = fracDiff(nasdaq_2017, nas[0], 0.05)
  gold_frac = fracDiff(gold_2017, gol[0], 0.05)
  print('Correlacao de Pearson BTC x NASDAQ-100 a partir de 2017:', btc_frac.corrwith(nasdaq_frac[nasdaq_frac.columns[0]]))
  print('Correlacao de Pearson BTC x ouro a partir de 2017:', btc_frac.corrwith(gold_frac[gold_frac.columns[0]]))
  print('Correlacao de Spearman BTC x NASDAQ-100 a partir de 2017:', btc_frac.corrwith(nasdaq_frac[nasdaq_frac.columns[0]], method = 'spearman'))
  print('Correlacao de Spearman BTC x ouro a partir de 2017:', btc_frac.corrwith(gold_frac[gold_frac.columns[0]], method = 'spearman'))

  print('')

  bit = adf_curve(btc_price_2020)
  nas = adf_curve(nasdaq_2020)
  gol = adf_curve(gold_2020)

  print('BTC', bit)
  print('NASDAQ', nas)
  print('Ouro',gol)

  btc_frac = fracDiff(btc_price_2020, bit[0], 0.05)
  nasdaq_frac = fracDiff(nasdaq_2020, nas[0], 0.05)
  gold_frac = fracDiff(gold_2020, gol[0], 0.05)
  print('Correlacao de Pearson BTC x NASDAQ-100 a partir de 2020:', btc_frac.corrwith(nasdaq_frac[nasdaq_frac.columns[0]]))
  print('Correlacao de Pearson BTC x ouro a partir de 2020:', btc_frac.corrwith(gold_frac[gold_frac.columns[0]]))
  print('Correlacao de Spearman BTC x NASDAQ-100 a partir de 2020:', btc_frac.corrwith(nasdaq_frac[nasdaq_frac.columns[0]], method = 'spearman'))
  print('Correlacao de Spearman BTC x ouro a partir de 2020:', btc_frac.corrwith(gold_frac[gold_frac.columns[0]], method = 'spearman'))
  return

if __name__ == '__main__':
  print_correlation()

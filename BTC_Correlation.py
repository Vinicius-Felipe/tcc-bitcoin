import pandas as pd
import numpy as np


def log_return(series):
  series_log = np.log(series).diff()
  series_log = series_log.dropna()
  return series_log

def period(series, year):
  series_period = series[series.index.year >= year]
  return series_period

# Pre processing data

btc = pd.read_csv('btc.csv')
btc = btc.rename(columns={"time": "date"})
btc = btc.set_index('date')
btc_price = pd.DataFrame(btc['PriceUSD'], index = btc.index)
btc_price = btc_price.dropna()
btc_price.index = pd.to_datetime(btc_price.index)
btc_price_log = log_return(btc_price)
btc_price_log_2017 = period(btc_price_log, 2017)
btc_price_log_2020 = period(btc_price_log, 2020)

nasdaq = pd.read_csv('nasdaq-100.csv', on_bad_lines='skip')
nasdaq = nasdaq.set_index('date')
nasdaq.index = pd.to_datetime(nasdaq.index)
nasdaq_log = log_return(nasdaq)
nasdaq_log = nasdaq_log[nasdaq_log.columns[0]]

gold = pd.read_csv('gold.csv')
gold = gold.rename(columns={"Date": "date"})
gold = gold.set_index('date')
gold.index = pd.to_datetime(gold.index)
gold['Price'] = gold['Price'].str.replace(',', '').astype(float)
gold_log = log_return(gold['Price'])

# Print results


def print_correlation():

  print('Correlacao de Pearson BTC x NASDAQ-100 período inteiro:', btc_price_log.corrwith(nasdaq_log)[0])
  print('Correlacao de Spearman BTC x NASDAQ-100 período inteiro:', btc_price_log.corrwith(nasdaq_log, method = 'spearman')[0])
  print('Correlacao de Pearson BTC x ouro período inteiro:', btc_price_log.corrwith(gold_log)[0])
  print('Correlacao de Spearman BTC x ouro período inteiro:', btc_price_log.corrwith(gold_log, method = 'spearman')[0])

  print('')

  print('Correlacao de Pearson BTC x NASDAQ-100 a partir de 2017:', btc_price_log_2017.corrwith(nasdaq_log)[0])
  print('Correlacao de Spearman BTC x NASDAQ-100 a partir de 2017:', btc_price_log_2017.corrwith(nasdaq_log, method = 'spearman')[0])
  print('Correlacao de Pearson BTC x ouro a partir de 2017:', btc_price_log_2017.corrwith(gold_log)[0])
  print('Correlacao de Spearman BTC x ouro a partir de 2017:', btc_price_log_2017.corrwith(gold_log, method = 'spearman')[0])

  print('')

  print('Correlacao de Pearson BTC x NASDAQ-100 a partir de 2020:',  btc_price_log_2020.corrwith(nasdaq_log)[0])
  print('Correlacao de Spearman BTC x NASDAQ-100 a partir de 2020:',btc_price_log_2020.corrwith(nasdaq_log, method = 'spearman')[0])
  print('Correlacao de Pearson BTC x ouro a partir de 2020:', btc_price_log_2020.corrwith(gold_log)[0])
  print('Correlacao de Spearman BTC x ouro a partir de 2020:',  btc_price_log_2020.corrwith(gold_log, method = 'spearman')[0])
  return

if __name__ == '__main__':
  print_correlation()
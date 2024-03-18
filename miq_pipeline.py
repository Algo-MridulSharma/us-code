"""
Contact: psar@algoanalytics.com
Company: AlgoAnalytics Pvt. Ltd.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import config
import json
from period_functions import *
import warnings
import miqp as miqp
import os
import time
import shm_functions
import random

#try:
import config

# except:
#   import src.config as config



def generate_stocks_dataframe(stocks:list,start_date:dt.datetime,end_date:dt.datetime):
  #TODO: add case to handle niftybees.ns inclusion, drop na rows for case 5
  stocks_historical_data_path=config.stocks_historical_data_src
  if os.path.exists(stocks_historical_data_path):
    historical_stocks_data = pd.read_csv(stocks_historical_data_path).set_index('Date')
    
    if not isinstance(historical_stocks_data.index, pd.DatetimeIndex):
      historical_stocks_data.index = pd.to_datetime(historical_stocks_data.index)
    
    historical_stocks_data.index = historical_stocks_data.index.strftime('%Y-%m-%d')
    historical_stocks_data.index = pd.to_datetime(historical_stocks_data.index)
    tmp_stocks=stocks[:]
    stocks=[item for item in tmp_stocks if item in historical_stocks_data.columns]

    given_stocks_dataframe = historical_stocks_data[stocks]
    given_stocks_dataframe=given_stocks_dataframe.ffill()
    return given_stocks_dataframe.loc[start_date:end_date]
  
def rolling_window_returns(tickers,date,period,window_size=10):
  sd,ed=get_start_end_from_date(date,period)
  start_date=sd.strftime('%Y-%m-%d')
  end_date=ed.strftime('%Y-%m-%d')
  column_suffix=""
  df=generate_stocks_dataframe(tickers,start_date,end_date)
  log_df=shm_functions.log_returns(df,df.columns.to_list(),log_return_column_suffix=column_suffix)
  window_df=shm_functions.non_overlap_rolling_window(log_df,df.columns.to_list(),window_size=window_size,daily_log_col_suffix=column_suffix)
  return window_df

def calculate_annualized_volatility(stock_data,month,y):
    """
    Calculate annualized volatility of stock data
    """
    volatility={}
    for stock in stock_data:
       df=pd.DataFrame()
       df['Returns']=stock_data[stock].pct_change()
       daily_volatility=df['Returns'].std()
       annualized_volatitlity=daily_volatility*np.sqrt(252)
       volatility[stock]=annualized_volatitlity
    
    file_path = "new_backtesting/filter3_stats.json"

# Load existing data if the file exists
    try:
        with open(file_path, 'r') as r:
            data = json.load(r)
    except FileNotFoundError:
        data = {}

    # Add the new month's data to the existing data
    data[y][month] = volatility

    # Save the updated data back to the file
    with open(file_path, 'w') as w:
        json.dump(data, w, indent=4)

def variance_optim_pipeline_modified(tickers,window_size,n_stocks,date):
    """
    This function performs a variance optimization pipeline.
    It calculates the covariance matrix, performs portfolio optimization,
    and returns the selected assets based on the optimization results.
    """

    period=-5

    df_10_day_log_returns= rolling_window_returns(tickers,date,period,window_size)

    #n_day_returns_df =  rolling_window_returns_5y(tickers,window_size,month)##replace this with required stocks 10 day returns
    covariance_matrix = np.cov(df_10_day_log_returns,rowvar=False) #rowvar False - log returns in columns
    tickers=list(df_10_day_log_returns.columns) #length of total assets

    stddevs = np.sqrt(np.diag(covariance_matrix)) #list of standard deviations of each stock
    average_std_dev = np.sqrt(np.mean(np.diag(covariance_matrix))) #average standard deviation of returns of all stocks
    average_std_dev_list = [average_std_dev]*3
    correlation_matrix = covariance_matrix / np.outer(stddevs, stddevs)

    mod_covariance_matrix = correlation_matrix * average_std_dev * average_std_dev 
                                                          
    
    #optimizer
    allocations = miqp.portfolio_optim(mod_covariance_matrix,n_stocks,len(tickers))
    allocation_bools = (allocations == 1)

    selected_assets = df_10_day_log_returns.columns[
        allocation_bools].to_list()
    return selected_assets

def variance_optim_pipeline_max_sharpe_ratio_modified_covariance(tickers,window_size,n_stocks,date):
  """
  sharpe ~ returns/std dev
  returns for 1 year data
  std dev for 5 year data using modified covariance
  top n selection for weights
  """

  #path where 5yr rolling window returns are saved for all the Filter 2 tickers for a month:-
  year_5_returns_df = rolling_window_returns(tickers,date,period=-5,window_size=window_size)
  year_1_returns_df = rolling_window_returns(tickers,date,period=-1,window_size=window_size)

  final_tickers=tickers[:]

  expected_returns = year_1_returns_df.mean().to_numpy()

  covariance_matrix = np.cov(year_5_returns_df,rowvar=False) # ddof = 1


  stddevs = np.sqrt(np.diag(covariance_matrix))
  average_std_dev = np.sqrt(np.mean(np.diag(covariance_matrix)))
  average_std_dev_list = [average_std_dev]*3
  correlation_matrix = covariance_matrix / np.outer(stddevs, stddevs)
  mod_covariance_matrix = correlation_matrix * average_std_dev * average_std_dev
    

  allocations = miqp.portfolio_sharpe_max_iterative_scipy(expected_returns,
                                                      mod_covariance_matrix,
                                                      n_stocks,
                                                      len(final_tickers))
  allocation_bools = (allocations == 1)

  selected_assets = year_5_returns_df.columns[
      allocation_bools].to_list()
  return selected_assets 





if __name__ == "__main__":
  universe3 = {}
  not_found = list()
  for month in config.months[:]:
    days_dict= config.trading_days_by_month[month]
    # for trading_day in config.trading_days[:1]:
    date=days_dict["1"]
    with open(f"shm_results/stocks-list-65/{month}.json",'r') as e:
      buying_data=json.load(e)
      universe2=buying_data[date]['Filter 2 Stocks etf']
      if(len(universe2) > 100):
        random.shuffle(universe2)
        universe2 = universe2[:99]
        
      # universe3[month]=variance_optim_pipeline_max_sharpe_ratio_modified_covariance(universe2,
      #                                                                         window_size=config.days,
      #                                                                         n_stocks=config.num_stocks,
      # date=date)
      try:
        universe3[month]=variance_optim_pipeline_modified(universe2, window_size=config.days, n_stocks=config.num_stocks, date=date)
      except:
        not_found.append(month)


      print(not_found)  
      out_file = open('./U3-final_selection/final_selection-65-etf.json', "w") 
      json.dump(universe3, out_file, indent=4)
    

         
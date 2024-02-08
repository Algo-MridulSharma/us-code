#import custom modules
from period_functions import *
# from filters import *
from miq_pipeline import *
import config
import timeit

#import standard modules
import numpy as np
import pandas as pd
import warnings
import json
import time
import math
import os
import glob
import datetime as dt
from datetime import timedelta,datetime

import re
import joblib
from sklearn.linear_model import LinearRegression
import calendar

#data fetch
import yfinance as yf


def empty_temp_folder():
    path = 'temp/'
    csv_files = glob.glob(os.path.join(path, '*.csv'))
    xlsx_files = glob.glob(os.path.join(path, '*.xlsx'))
    for file in csv_files:
      os.remove(file)
    for file in xlsx_files:
      os.remove(file)

def swap_elements(lst, n):
    if n >= len(lst):
        return lst 
    lst[0], lst[n] = lst[n], lst[0]
    return lst

def generate_log_returns(df,index_tickers):
    """The function calculates the log returns for index and stocks in given dataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing Historical Data of Stocks
        index_tickers (list): Index+ Stock tickers

    Returns:
        pd.DataFrame: Log returns Dataframe
    """
    for ticker in index_tickers:
        ld=df[ticker]
        if isinstance(ld, pd.DataFrame) and len(ld.columns)>1: #handles the case where ld is a DataFrame and not a Pandas series
            log_returns = np.log(ld.iloc[:, 0]) - np.log(ld.iloc[:, 0].shift(1)) 
        else:
            log_returns = np.log(df[ticker]) - np.log(df[ticker].shift(1)) #current - previous

        df = pd.concat([df, log_returns.rename(ticker + '_log_returns')], axis=1)
    return df.iloc[1:]

def data_for_specific_time_period(time_range,df):
    """
    This function filters the given dataFrame and returns dataframe for required time period.
    """
    last_date_from_df = df.iloc[-1]
    last_date_1 = last_date_from_df.name
    #last_date_1=last_date_from_df
    start_date_1 = last_date_1 - timedelta(days=365.25)#latest
    start_date_2 = start_date_1 - timedelta(days=365.25)
    start_date_3 = start_date_2 - timedelta(days=365.25)
    start_date_4 = start_date_3 - timedelta(days=365.25)
    start_date_5 = start_date_4 - timedelta(days=365.25)#oldest
    #print(f"For {time_range}-------------------------\n")
    
    if time_range == "1 Year":
        last_data = df.iloc[-1]
        last_date = last_data.name
        start_date = start_date_1
        filtered_df = df[df.index >= start_date]
        
        return filtered_df
    if time_range == "2 Year":

        last_date = start_date_1
        start_date = start_date_2
        filtered_df = df[(df.index >= start_date) & (df.index < last_date)]

        return filtered_df
    if time_range == "3 Year":

        last_date = start_date_2
        start_date = start_date_3
        

        filtered_df = df[(df.index >= start_date) & (df.index < last_date)]
        
        return filtered_df
    if time_range == "4 Year":

        last_date = start_date_3
        start_date = start_date_4


        filtered_df = df[(df.index >= start_date) & (df.index < last_date)]
        return filtered_df
    if time_range == "5 Year":

        last_date = start_date_4
        start_date = start_date_5
    
        filtered_df = df[(df.index >= start_date) & (df.index < last_date)]
        return filtered_df

def get_sharpe_ratio(model,zero_mean_residuals):
    """ Functions calculates the Sharpe ratio for a given stock on the basis of provided Alpha,
        

    Args:
        model (tuple): Alpha - Excess return on investment above risk free return. 
        zero_mean_residuals (tuple): deviation of predicted stock returns from actual stock returns. Volatility of the stock

    Returns:
        float: Sharpe Ratio
        """
    C = model[0] #Alpha - Excess return on investment above risk free return.
    Sharpe_ratio = C / (np.std(zero_mean_residuals)) #excess return of stock per unit risk
    return Sharpe_ratio

def get_sharpe_ratio_modified(model,zero_mean_residuals):
    """ Functions calculates the Modifed sharpe ratio for a given stock on the basis of provided Alpha.
     Multiplying by volatility rather than dividing by the same in standard formula
        

    Args:
        model (tuple): Alpha - Excess return on investment above risk free return. 
        zero_mean_residuals (tuple): deviation of predicted stock returns from actual stock returns. Volatility of the stock

    Returns:
        float: Sharpe Ratio
        """
    C = model[0] #Alpha - Excess return on investment above risk free return.
    Volatility = np.std(zero_mean_residuals)
    mod_sharpe_ratio = C * Volatility #excess return of stock per unit risk
    return mod_sharpe_ratio, Volatility

def linear_regression_model(df, index, index_ticker, window):
    """
    The function calculates the values of Alpha, Beta and Residuals for a given stock wrt index.

    stock = index(β) + α

    β - Sensitivity of the stock wrt to index
    α - Excess returns of the stock when compared to what would be expected based on its
       relation with the index (Beta)
    
    """
    
    df_temp = df[[index+window+'_log_returns', index_ticker+window+'_log_returns']]
    # df_temp = df.dropna() #to drop NA values in old code

    if df_temp.isna().all().all() or df_temp.shape[0]<10:
        return [np.nan,np.nan],[np.nan,np.nan]
    x = df_temp[index+window+'_log_returns'].values.reshape(-1, 1) #index's Log returns
    y = df_temp[index_ticker+window+'_log_returns'].values.reshape(-1, 1) #Stocks' Log Returns

    regressor = LinearRegression()
    regressor.fit(x, y)
    intercept = regressor.intercept_[0]
    slope = regressor.coef_[0][0]
    y_pred = regressor.predict(x)
    residuals = y - y_pred #actual stock returns - predicted stock returns
    return [intercept, slope], residuals

def set_default_results(days):
    """Function returns an ordered empty Dataframe

    Args:
        days (int): log returns days

    Returns:
        pd.Dataframe: empty dataframe in desired order
    """
    return pd.DataFrame(columns=['Name of the Stock', 'Trading Days', 
                               f'β{days}', f'α{days}', f'ShM{days}',f'Volatality{days}',f'mod_ShM{days}'])

def new_results_append_modified(index_ticker,ticker,models,length,df,resid,days):
    """ Calculate the modified sharpe ratio for turn around stories and append 
    the Alpha, Beta, ShM, Volatility and Mod_sharpe_ratio values to "results" dataFrame.

    Args:
        index_ticker (str): index
        ticker (str): Stock ticker
        models (tuple): Containing slope and intercept.
        length (int): length(results dataframe)
        df (pd.Dataframe): stocks historical dataframe
        resid (np.array): residuals
        days (int): log return days

    Returns:
        pd.Dataframe: Dataframe containing stock wise Alpha, beta & ShM
    """

    df_copy = df.copy()
    df_copy = df_copy[[f"{ticker}_{days}d_log_returns"]]
    df_copy = df_copy.dropna()
    mod_sharpe_ratio=get_sharpe_ratio_modified(models[0],resid[0]) #mod_ShM, C
    return pd.DataFrame({
        'Name of the Stock': ticker,
        'Trading Days': df_copy.shape[0],
        f'β{days}': models[0][1],
        f'α{days}': models[0][0],
        f'ShM{days}': get_sharpe_ratio(models[0],resid[0]),
        f'Volatality{days}': mod_sharpe_ratio[1],
        f'mod_ShM{days}': mod_sharpe_ratio[0]
    }, index=[length])

def new_results_append(index_ticker,ticker,models,length,df,resid,days):
    """ Calculate the sharpe ratio and append the Alpha, Beta and ShM values to "results" dataFrame.

    Args:
        index_ticker (str): index
        ticker (str): Stock ticker
        models (tuple): Containing slope and intercept.
        length (int): length(results dataframe)
        df (pd.Dataframe): stocks historical dataframe
        resid (np.array): residuals
        days (int): log return days

    Returns:
        pd.Dataframe: Dataframe containing stock wise Alpha, beta & ShM
    """

    df_copy = df.copy()
    df_copy = df_copy[[ticker+"_log_returns"]]
    df_copy = df_copy.dropna()
    
    return pd.DataFrame({
        'Name of the Stock': ticker,
        'Trading Days': df_copy.shape[0],
        f'β{days}': models[0][1],
        f'α{days}': models[0][0],
        f'ShM{days}': get_sharpe_ratio(models[0],resid[0])
    }, index=[length])

def get_results_regression(index,index_ticker,df,results,reg,days):
    """The function chooses the type of regression for calculating Alpha, beta and ShM

    Args:
        index (str): Index whose stocks are being checked for.
        index_ticker (list): Member stocks of index.
        df (pd.Dataframe): Dataframe containing historical prices.
        results (pd.Dataframe): empty dataframe having ordered-columns
        reg (str): "LR"
        days (int): "10"

    Returns:
        pd.DataFrame: Dataframe containing stock wise Alpha, Beta and ShM
    """

    if reg == "LR":

        #print("Linear Regression ",days,'d')
        for ticker in index_ticker:        
            model,residuals = linear_regression_model(df,index,ticker,f"_{days}d")
            if not np.isnan(model[0]) and not np.isnan(residuals[0]):

                # new_row = new_results_append(index_ticker,ticker,[model],len(results),df,[residuals],days)
                # results = pd.concat([results, new_row])

                new_row_mod = new_results_append_modified(index_ticker,ticker,[model],len(results),df,[residuals],days)
                results = pd.concat([results, new_row_mod])
        results['mod_ShM_1000']=results[f'Volatality{days}']*results[f'α{days}']*1000
    else:
        print("Linear Regression is the best one yet, why not choose it?")

    return results


def generate_file_name(month:str,year:str):
    """
    The function generates name of the index membership file.
    Input: Jan 2017
    Output: data/index-constituents/main/indices-data-Q1-2017.json
    Args:
        month (str): Month for which the stock memberships is to be searched
        year (str): Year for the same

    Returns:
        Str : "data/index-constituents/main/indices-data-Q1-2017.json"
    """

    q=config.monthsdict[config.MAPPING_DICT[month]]
    file_name = f"/tmp/indices-data-{q}-{year}.json"
    return file_name

def get_index_name(index_symbol):
    for item in config.nifty_symbols:
        # print(item)
        if item['Trading_Symbol'].upper() == index_symbol.upper():
            return item['Index_Name'].upper()
     
def get_index_symbol(long_name):

    for item in config.nifty_symbols:
        # print(item)
        if item['Index_Name'].upper() == long_name.upper():
            return item['Trading_Symbol']

def generate_dataframe_regression(start_date,end_date,index,stocks):
    
    """
    This function generates a dataframe for the given time range, stocks and index.
    """
    stocks_not_from5y=[]

    df = pd.DataFrame()
    historical_path_stocks=config.stocks_historical_data_src
    df_s=pd.read_csv(historical_path_stocks).set_index('Date')

    historical_path_indices=config.indices_historical_data_src
    df_i=pd.read_csv(historical_path_indices).set_index('Date')
    cols = df_i.columns.to_list()

    miss_stocks=[stock for stock in stocks if stock not in df_s.columns]
    m_stocks=[]
    if len(miss_stocks) !=0:
        m_stocks.extend(miss_stocks)
        print(index)
        print(f"Missing stocks - {m_stocks}")
        stocks = [item for item in stocks if item not in m_stocks]


    df_s=df_s[stocks]
    df_s.index = pd.to_datetime(df_s.index)
    df_s.index = df_s.index.strftime('%Y-%m-%d')
    df_s.index = pd.to_datetime(df_s.index)
    f_df_s=df_s.loc[start_date:end_date]

    # YFinance give np.NaN values for all stocks for weekday-holidays
    rows_with_all_nan = f_df_s[f_df_s.isna().all(axis=1)]
    weekday_holidays = rows_with_all_nan.index
    #print(f"weekdday holidays : {len(weekday_holidays)}")
    f_df_s=f_df_s.drop(weekday_holidays)

    #print("Dropping holidays (weekday): \n",list(weekday_holidays))
    
    final_stocks=[]
    for stock in stocks:
        sdf=f_df_s[[stock]]
        sdf=sdf.loc[:, ~sdf.columns.duplicated()]

        try:
            nan_count = sdf.isna().sum()
            NaN_ratio=nan_count/len(f_df_s[[stock]])
            #choose the stocks whose whole 5 year data is available
            if (NaN_ratio == 0): #dont choose stocks which have np.NaN value
                #print(f"{stock} in existence from 5 years")             
                final_stocks.append(stock)
            else:
                stocks_not_from5y.append(stock)
        except Exception as b:
            nan_count = sdf.isna().sum().iloc[0]
            first_isna=pd.isna(sdf[stock].iloc[0])
            NaN_ratio=nan_count/len(f_df_s[[stock]])
            #choose the stocks whose whole 5 year data is available

            if (NaN_ratio == 0): #dont choose stocks which have np.NaN value
                #print(f"{stock} in existence from 5 years")             
                final_stocks.append(stock)
            elif (nan_count < 20) and not first_isna:
                final_stocks.append(stock) #case where data is missing for companies in between time range. 
            else:
                stocks_not_from5y.append(stock)


     

    try:
        df_i.index = pd.to_datetime(df_i.index, format='%Y-%m-%d').date
        f_df_i=df_i.loc[start_date:end_date]
        columns_with_nan = f_df_i.columns[f_df_i.isna().all()].tolist()
        if index in columns_with_nan:
            print("No data for index")
            return pd.DataFrame()#return empty df if data for index not there
        

    except Exception as e:
        print(e)

    
    df = f_df_s.copy()
    try:
        index=index.upper()
        df[index]=f_df_i[index]
    except:
        try:
            index_name=get_index_name(index)
            df[index]=f_df_i[index_name]
        except:
            return pd.DataFrame()
    

    final_stocks=list(set(final_stocks))
    final_stocks.insert(0,index)
    df=df[final_stocks] #dataframe of required stocks and indices
    final_df = df.dropna(subset=[index]) #drop the dates which have NaN values acc to index since data is from NSE
    warnings.filterwarnings('ignore')

    final_df.ffill(inplace=True) ## Forward fill for stocks incase data is there for index but not for stock
    
    # Identify and drop duplicate columns
    final_df = final_df.loc[:, ~final_df.columns.duplicated(keep='first')]
    return final_df

def generate_df_stock_membership(start_date,end_date,index,stocks,quarter):
    df = pd.DataFrame()
    #historical_path_stocks=f"data/stocks-data.csv"
    historical_path_stocks=config.stocks_historical_data_src
    historical_path_indices=config.indices_historical_data_src

    df_i=pd.read_csv(historical_path_indices).set_index('Date')
    df_s=pd.read_csv(historical_path_stocks).set_index('Date')
    cols = df_i.columns.to_list()

    df_s.index = pd.to_datetime(df_s.index)
    df_s.index = df_s.index.strftime('%Y-%m-%d')
    df_s.index = pd.to_datetime(df_s.index)
    f_df_s=df_s.loc[start_date:end_date]

    try:
        df_i.index = pd.to_datetime(df_i.index, format='%Y-%m-%d').date
        f_df_i=df_i.loc[start_date:end_date]
    except Exception as e:
        print(e)

    df_i.dropna(axis=1,inplace=True) #dropping indices with no available data
    df = f_df_s.copy()
    df[index]=f_df_i[index]
    stocks.insert(0,index)
    df=df[stocks] #dataframe of required stocks and indices
    final_df = df.dropna(subset=[index]) #drop the dates which have NaN values.
    empty_columns = list(final_df.columns[final_df.isna().any()])
    final_df=final_df.drop(columns=empty_columns) # removes the stocks which haven't existed for the entire given time period
    return final_df

def log_returns(df:pd.DataFrame(), 
                tickers:list = None,
                log_return_column_suffix:str = "_log_returns"):
    """Calculates Daily log return and adds it as a column to existing df with 
        a suffix

    Args:
        df (pd.DataFrame or pd.Series if tickers = None): Sorted df with respect 
            to Date Column (Ascending) with stock and ticker close prices in columns.
        tickers (list): list of tickers to process.
        log_return_column_suffix (str, optional): _description_. Defaults to "_log_returns".

    Returns:
        (pd.DataFrame):
    """
    total=df.columns.to_list()
    df_copy = df.copy()

    
    tickers_with_suffix = [ticker + log_return_column_suffix 
                        for ticker in tickers]
    # df_copy.to_csv('./checkdf_copy.csv')
    
    # convering all string values (1,000) to float 
    for col in df_copy.columns:
        if pd.api.types.is_float_dtype(df_copy[col]):
            continue;
        else:
            df_copy[col] = (df_copy[col].str.replace(",", "")).astype(float) # handles non-numeric values            
   
    df_copy[tickers_with_suffix] = np.log(df_copy[tickers]) - np.log(df_copy[tickers].shift(1))
    

    df_copy = df_copy.iloc[1:] #selecting the dataframe from 2nd row onwards
    return df_copy

def non_overlap_rolling_window(orginal_df, 
                               tickers, 
                               window_size=10,
                               date_col="Date", 
                               daily_log_col_suffix = "_log_returns"):
    """
    This function calculates non-overlapping rolling windows of log returns for a given DataFrame.
    
    Parameters:
        orginal_df (DataFrame): The original DataFrame containing the data.
        tickers (list): A list of tickers for which to calculate the log returns.
        date_col (str): The name of the column containing the date values. Default is "Date".
        window_size (int): The size of the rolling window. Default is 10.
        daily_log_col_suffix (str): The suffix to add to the column names of the log returns. Default is "_log_returns".
    
    Returns:
        DataFrame: A DataFrame containing the non-overlapping rolling windows of log returns.
    """
    df = orginal_df.copy()
    df.reset_index(inplace=True)
    # Create groups based on the specified window size
    df["groups"] = df.index // window_size #quotient 
    # Find the last group and check if it has enough data points
    last_group = df["groups"].max()
    last_group_data = df[df["groups"] == last_group]
    
    # Exclude the last group if it has fewer than half of the required data points
    if last_group_data[date_col].count() < window_size / 2:
        df = df[df["groups"] != last_group].copy()

    # Create the aggregation dictionary
    agg_dict = {ticker + daily_log_col_suffix: 'sum' for ticker in tickers}
    agg_dict[date_col] = "min"

    # Calculate returns for each group and include the minimum date
    non_overalapping_rolling_window = df.groupby("groups").agg(
        agg_dict
        ).reset_index()
    
    # Drop the "groups" column, set the index, and rename the columns
    non_overalapping_rolling_window = non_overalapping_rolling_window.drop(
        "groups",
        axis=1)
    non_overalapping_rolling_window = non_overalapping_rolling_window.set_index(
        date_col
        )
    if daily_log_col_suffix != "":

        non_overalapping_rolling_window.columns = [
            column.replace(daily_log_col_suffix,"") + f"_{window_size}d_log_returns" 
            for column in non_overalapping_rolling_window.columns
            ]
    else:
        print("Generating rolling window for MIQP")
    return non_overalapping_rolling_window


if __name__ == "__main__":
    print("Write tests here")
            

from jugaad_data.nse import stock_df
import yfinance as yf
from tqdm import tqdm
from period_functions import *
import config
#from config import nifty_data,nifty_etf,nse
import os
import json
import pandas as pd

#load stocks data

df_s=pd.read_csv(config.stocks_historical_data_src).set_index('Date')
# df_i=pd.read_csv("S&P_500_data.csv").set_index('Date')
df_i=pd.read_csv("S&P_500_ETF_data.csv").set_index('Date')


df_s.index = pd.to_datetime(df_s.index)
df_s.index = df_s.index.strftime('%Y-%m-%d')
df_s.index = pd.to_datetime(df_s.index)
df_s.ffill(inplace=True)


df_i.index = pd.to_datetime(df_i.index, format='%Y-%m-%d').date



def stock_historical_price(stock_symbol, start_d, end_d):
    #TODO: the file is getting opened again and again, instead open it once only while loading in config
    f_df_s=df_s.loc[start_d:end_d]
    return f_df_s[stock_symbol]

def index_historical_price(stock_symbol, start_d, end_d):
    #TODO: the file is getting opened again and again, instead open it once only while loading in config
    f_df_i=df_i['Close'].loc[start_d:end_d]
    return f_df_i

def get_return(stock_symbol:str,st,ed,start_d,end_d)->float:
    """This function calculates the return % of stock for a given time period.

    Args:
        stock_symbol (str): _description_
        st (_type_): _description_
        ed (_type_): _description_
        start_d (_type_): _description_
        end_d (_type_): _description_

    Returns:
        Float: Return %
    """
    if stock_symbol.endswith('500'):
        index_data=index_historical_price(stock_symbol,st,ed)
        start_price = round(index_data.iloc[0],2)
        end_price = round(index_data.iloc[-1],2)
    else:
        stock_data = stock_historical_price(stock_symbol, start_d, end_d)
        start_price = round(stock_data.iloc[0],2)
        end_price = round(stock_data.iloc[-1],2)
        
        
        

    
    #Calculate the input quarter return
    returns = round(((end_price - start_price) / start_price) * 100,2)
    #print(f"Price on {str(start_d)} = {start_price} & Price on {str(end_d)} = {end_price} for {stock_symbol} returns = {returns}")

    return returns

def add_1_day_return(df,today):
    returns_1_day=[]
    close_prices=[]
    print('Calculating 1 day returns for ',today)
    for stock in df['Stock']:
        st,ed = get_start_end_from_date(today,period=-0.5)
        start_d= st.strftime('%Y-%m-%d')
        end_d = ed.strftime('%Y-%m-%d')
        stock_data = stock_historical_price(stock, start_d, end_d)
        start_price = round(stock_data.iloc[-2],2)
        end_price = round(stock_data.iloc[-1],2)
        returns = round(((end_price - start_price) / start_price) * 100,2)
        returns_1_day.append(returns)
        close_prices.append(end_price)

    
    df['1_day_return']=returns_1_day
    df['prev_close']=close_prices
    return df

def add_1_year_return(df,today):
    returns_1_year=[]
    print('Calculating 1 Year returns for ',today)
    for stock in tqdm(df['Stock'].to_list(),desc="Filter 2",unit="stock"):
        st,ed = get_start_end_from_date(today,period=-1)
        start_d= st.strftime('%Y-%m-%d')
        end_d = ed.strftime('%Y-%m-%d')
        returns_1_year.append(get_return(stock,st,ed,start_d,end_d))
    df['1_year_return']=returns_1_year
    return df

def add_1_month_return(df,today):
    returns_1_month=[]
    st,ed = get_start_end_from_date(today,period=-(1/12))
    start_d= st.strftime('%Y-%m-%d')
    end_d = ed.strftime('%Y-%m-%d')
    print(f'Calculating 1 month returns for {today}; start:{start_d}; end:{end_d}')
    for stock in tqdm(df['Stock'].to_list(),desc="1 month return",unit="stock"):
        returns_1_month.append(get_return(stock,st,ed,start_d,end_d))
    df['1_month_return']=returns_1_month
    return df

def add_15_day_return(df,today):
    returns_15_day=[]
    st,ed = get_start_end_from_date(today,period=-(1/24))
    start_d= st.strftime('%Y-%m-%d')
    end_d = ed.strftime('%Y-%m-%d')
    print(f'Calculating 15 day returns for {today}; start:{start_d}; end:{end_d}')
    for stock in tqdm(df['Stock'].to_list(),desc="15 day return",unit="stock"):
        returns_15_day.append(get_return(stock,st,ed,start_d,end_d))
    df['15_day_return']=returns_15_day
    return df

def add_sp500_return(symbol,today,period=-1):
    st,ed = get_start_end_from_date(today,period)
    start_d = st.strftime('%Y-%m-%d')
    end_d = ed.strftime('%Y-%m-%d')
    return get_return(symbol,st,ed,start_d,end_d)


def add_index_difference(df,index,today):
    st,ed = get_start_end_from_date(today,period=-1)
    start_d = st.strftime('%Y-%m-%d')
    end_d = ed.strftime('%Y-%m-%d')
    index_return=get_return(index,st,ed,start_d,end_d)
    df[f'1_year_{index}_return']=index_return
    try:
        df[f'return_diff_{index}']=df['1_year_return']-df[f'1_year_{index}_return']
    except:
        df[f'return_diff_{index}']=df['52_week_return']-df[f'1_year_{index}_return']
    return df

def add_custom_return(df,today,period,df_key):
    returns=[]
    print(f'Calculating {period*12} months returns for ',today)
    for stock in tqdm(df['Stock'].to_list(),desc="Filter 2 etf",unit="stock"):
        st,ed = get_start_end_from_date(today,period=period)
        start_d= st.strftime('%Y-%m-%d')
        end_d = ed.strftime('%Y-%m-%d')
        returns.append(get_return(stock,st,ed,start_d,end_d))
    df[df_key]=returns
    return df



def add_sp500_difference(df,today):
    period=-1
    final_df=add_custom_return(df,today,period=period,df_key="1_year_return")
    final_df['1_year_sp500_return']=add_sp500_return("SP500",today,period=period)
    final_df[f'1y_return_diff_sp500']=final_df['1_year_return']-final_df[f'1_year_sp500_return']
    return final_df

def generate_universe2(universe1,month,date):
    print("Generating Universe 2")
    # Convert to datetime object
    original_date = datetime.strptime(date, "%Y-%m-%d")

    # Format the datetime object as a string with leading zeros
    date1 = original_date.strftime("%Y-%m-%d")
    
    os.makedirs(f'universe-2/{month}',exist_ok=True)
    filter2_path=f'universe-2/{month}/{date}-universe-2.csv'
    # if os.path.exists(filter2_path):
    #     stock_universe=pd.read_csv(filter2_path)
    # else:
    stock_universe=add_sp500_difference(universe1,date) 
    stock_universe.to_csv(filter2_path,index=False)
    u2_stocks = stock_universe[stock_universe['1y_return_diff_sp500'] < 0]['Stock'].tolist()
    return u2_stocks
    
if __name__ =="__main__":
    for month in config.months:
        u1_json_path=f"shm_results/stocks-list-65/{month}.json"
        u1_dict=json.load(open(u1_json_path))
        date=config.trading_days_by_month[month]["1"]
        u1_stocks=u1_dict[date]['Filter 1 Stocks']
        # u1_stocks.append("SP500")
        universe1=pd.DataFrame({"Stock":u1_stocks})
        u2_stocks=generate_universe2(universe1,month,date)
        date_dict=u1_dict[date]
        date_dict['Filter 2 Stocks etf bad']=u2_stocks
        u1_dict[date]=date_dict
        json.dump(u1_dict,open(u1_json_path,'w'),indent=4)
    
from jugaad_data.nse import stock_df
import yfinance as yf
import period_functions as pf
from config import nifty_etf,stocks_historical_data_src
import pandas as pd


def stock_historical_price(stock_symbol, start_d, end_d):
    historical_path_stocks=stocks_historical_data_src
    df_s=pd.read_csv(historical_path_stocks).set_index('Date')
    df_s.index = pd.to_datetime(df_s.index)
    df_s.index = df_s.index.strftime('%Y-%m-%d')
    df_s.index = pd.to_datetime(df_s.index)
    f_df_s=df_s.loc[start_d:end_d]
    f_df_s.ffill(inplace=True)
    return f_df_s[stock_symbol]

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
    
    if stock_symbol=="NIFTYBEES.NS":
        stock_data = yf.download(stock_symbol, start=start_d, end=end_d)
        start_price = round(stock_data.iloc[0]['Adj Close'],2)
        end_price = round(stock_data.iloc[-1]['Adj Close'],2)
    else:
        stock_data = stock_historical_price(stock_symbol, start_d, end_d)
        start_price = round(stock_data.iloc[0],2)
        end_price = round(stock_data.iloc[-1],2)

    
    #Calculate the input quarter return
    returns = round(((end_price - start_price) / start_price) * 100,2)
    print(f"Price on {str(start_d)} = {start_price} & Price on {str(end_d)} = {end_price} for {stock_symbol} returns = {returns}")

    return returns

def add_1d_m_return(df,today):
    returns_1_day=[]
    close_prices=[]
    returns_monthly=[]
    print('Calculating 1 day, Month returns for ',today)
    for stock in df['Stock']:
        st,ed = pf.get_start_end_from_date(today,period=-1/12)
        start_d= st.strftime('%Y-%m-%d')
        end_d = ed.strftime('%Y-%m-%d')
        stock_data = stock_historical_price(stock, start_d, end_d)
        start_price = round(stock_data.iloc[-2],2)
        end_price = round(stock_data.iloc[-1],2)
        returns = round(((end_price - start_price) / start_price) * 100,2)
        start_price_m=round(stock_data.iloc[0],2)
        returns_m=round(((end_price - start_price_m) / start_price_m) * 100,2)
        returns_1_day.append(returns)
        returns_monthly.append(returns_m)
        close_prices.append(end_price)

    df['1_day_return']=returns_1_day
    df['prev_close']=close_prices
    df['1_month_return']=returns_monthly
    return df

def add_1_year_return(df,today):
    returns_1_year=[]
    print('Calculating 12 week returns for ',today)
    for stock in df['Stock']:
        st,ed = pf.get_start_end_from_date(today,period=-1)
        start_d= st.strftime('%Y-%m-%d')
        end_d = ed.strftime('%Y-%m-%d')
        returns_1_year.append(get_return(stock,st,ed,start_d,end_d))
    df['1_year_return']=returns_1_year
    return df

def add_nifty_return(nifty_etf:str,today)->tuple:
    """ 
    The function calculates returns for NIFTY ETF for a given period of time.

    Input:
    NIFTYBEES.NS,
    "2023-12-06"

    Output:
    7.8%

    Args:
        nifty_etf (str): preferrably "NIFTYBEES.NS"
        today (DateTime): Date from which the 1 year return is to be calculated

    Returns:
        tuple: Returns in %
    """
    st,ed = pf.get_start_end_from_date(today,period=-1)
    start_d = st.strftime('%Y-%m-%d')
    end_d = ed.strftime('%Y-%m-%d')
    return get_return(nifty_etf,st,ed,start_d,end_d)

def add_nifty_difference(df,today):
    final_df=add_1_year_return(df,today)
    nifty_return=add_nifty_return(nifty_etf,today)
    final_df['1_year_nifty_return']=nifty_return
    final_df['return_diff_nifty']=final_df['1_year_return']-final_df["1_year_nifty_return"]
    returns_df=add_1d_m_return(final_df.copy(),today)
    return returns_df


if __name__ == "__main__":
    import datetime as dt
    today = dt.date.today()
    st,ed = pf.get_start_end_from_date(today,period=-1)
    st,ed = pf.get_start_end_from_date(today,period=-1)
    start_d= st.strftime('%Y-%m-%d')
    end_d = ed.strftime('%Y-%m-%d')
    stock="SJVN.NS"
    get_return(stock,st,ed,start_d,end_d)
    pass
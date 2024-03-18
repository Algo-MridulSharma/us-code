import pandas as pd
import warnings
import numpy as np
import os,json
import yfinance as yf
from datetime import datetime, timedelta
import config
import backtest_config as btc
from period_functions import *
from config import parent_folder
from tqdm import tqdm

EXPERIMENTS=False
DETAILED_REPORT=True

cwd = os.getcwd()

exp="alpha_based"
parent_folder=parent_folder+"/meta" #+"/CS-F2+BX-F2"
#TODO: Change paths to fit with experiments and their folder structure.
investment_track_path=f"{parent_folder}/backtest-data-{exp}/track-investments"
os.makedirs(investment_track_path,exist_ok=True)
investment_track={}
        

STOCKS_DF=pd.read_csv(config.stocks_historical_data_src).set_index('Date') #stocks historical data


def generate_historical_data_df(start_date,end_date):
    """Generates historical DataFrame for Stocks

    Args:
        start_date (Str): Start Date for simulation
        end_date (Str): End date

    Returns:
        DataFrame: Stocks Historical Data
    """

    STOCKS_DF.index = pd.to_datetime(STOCKS_DF.index)
    STOCKS_DF.index = STOCKS_DF.index.strftime('%Y-%m-%d')
    STOCKS_DF.index = pd.to_datetime(STOCKS_DF.index)
    try:
        final_df=STOCKS_DF.loc[start_date:end_date]
    except:
        print("Problem with loading historical data")
    return final_df

def save_data_to_json(data, file_path):
    """
    Save data to a JSON file.

    Parameters:
    - data: The data to be saved (can be a dictionary, list, etc.).
    - file_path: The path where the JSON file will be saved.

    Returns:
    - None
    """
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

percent_return_dict={}

nifty_symbol= config.nifty_etf#+config.nse
num_stocks= config.num_stocks

def generate_results_df(data:list):
    from collections import defaultdict
    result_dict = defaultdict(list)
    for entry in data:
        date_key = entry["Date"]
        investment_value = entry["Investment"]
        result_dict[date_key].append(investment_value)

    # Modify the dictionary
    for date_key, investment_values in result_dict.items():
        # Add 1000 to the list only if needed
        investment_values.extend([1000] * max(0, 12 - len(investment_values)))

        # Calculate the portfolio value
        portfolio_value = sum(investment_values)

        # Update the dictionary with the new structure
        result_dict[date_key] = {
            "Investments": investment_values,
            "Final Investment": portfolio_value
        }

    dates = list(result_dict.keys())
    portfolio_values = [entry['Final Investment'] for entry in result_dict.values()]

    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Final Investment': portfolio_values
    })

    # Set the "Date" column as the index
    df.set_index('Date', inplace=True)

    df=df.sort_index()
    return df

def detailed_results_df(data:dict,investment_path,save=False):
    #print("Generating Detailed Dataframe")
    buyingperiod='all_jan'
    portfolio_dict=data[buyingperiod]
    suff,month=buyingperiod.split("_")
    portfolio_df = pd.DataFrame(list(portfolio_dict.items()), columns=['Date', f'{month.capitalize()}'])

    # Convert the 'Date' column to datetime type
    portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])

    # Set the 'Date' column as the index
    portfolio_df.set_index('Date', inplace=True)
    main_df=portfolio_df.copy()
    for buyingperiod in list(data.keys())[1:]:
        portfolio_dict=data[buyingperiod]
        suff,month=buyingperiod.split("_")
        portfolio_df = pd.DataFrame(list(portfolio_dict.items()), columns=['Date', f'{month.capitalize()}'])

        # Convert the 'Date' column to datetime type
        portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])

        # Set the 'Date' column as the index
        portfolio_df.set_index('Date', inplace=True)
        main_df=pd.merge(main_df, portfolio_df, on='Date', how='outer')
    main_df.fillna(1000,inplace=True)
    main_df['Final Investment'] = main_df.iloc[:, :].sum(axis=1) #compute the sum of total investment over 5 years
    if save: 
        main_df.to_csv(investment_path,index=True)
        print("Saved")
    return main_df[['Final Investment']]



def get_selected_stocks(month:str,trading_day:str,stocks_filter:str):
    """For a given Stock filter, this function returns stocks that should be bought\n
    for a buying date.

    Args:
        month (str): Ex: Sep 2022
        trading_day (str): 1st
        stocks_filter (str): Filter 1 Stocks

    Returns:
        list: list of stocks
    """
    days_dict= config.trading_days_by_month[month] #obtain the dict of investment dates
    stocks_list_path=f"{config.parent_folder}/shm_results/stocks-list-5/{month}.json"
    with open(stocks_list_path) as e:
        month_dict=json.load(e)

    if stocks_filter == "All Stocks": 
        return month_dict[stocks_filter]
    else:
        date=days_dict[trading_day] #get date on basis of trading day
        selected_stocks=month_dict[date][stocks_filter]
        if len(selected_stocks) == 0:
            raise Exception(f"0 Stocks chosen - {date}")
        return selected_stocks
    

def calculate_params(df,initial_investment=12000,investment_col="Final Investment",years=10):
    initial_capital = initial_investment
    print("Initial Capital: ",initial_capital)
    strategy_final_capital = df[investment_col].iloc[-1]
    print("Final Capital: ",round(strategy_final_capital,0))

    annualized_returns = round((((strategy_final_capital/ initial_capital) ** (1 /years)) - 1)*100,2)
    try:
        df.set_index('Date',inplace=True)
    except:
        pass
        
    investment_data = df[investment_col]
    df.index = pd.to_datetime(df.index)

    first_date = df.index[0].date()
    last_date = df.index[-1].date()
    trading_days = len(df.index)

    #calculate max drawdown
    cumulative_returns = (investment_data / investment_data.cummax()) - 1
    cr_ex=(investment_data / investment_data.cummax())
    drawdown = cumulative_returns.min()
    max_drawdown = round(-drawdown,2) 
    returns = investment_data.pct_change()
    risk_free_rate = 0.00
    trading_days_yearly = trading_days/years

    annualized_volatility = returns.std() * (trading_days_yearly ** 0.5)
    downside_returns = returns.copy()

    downside_returns[downside_returns > 0] = 0
    downside_volatility = downside_returns.std() * (trading_days_yearly ** 0.5)

    sharpe_ratio = round(((annualized_returns/100) - risk_free_rate) / annualized_volatility, 2)
    sortino_ratio = round(((annualized_returns/100)  - risk_free_rate) / downside_volatility, 2)
    calmar_ratio = round((annualized_returns/100) / max_drawdown,2)

    params_dict={}
    params_dict['Initial Capital']=initial_capital
    params_dict['Final Capital']=strategy_final_capital
    params_dict['Annualized Returns']=annualized_returns
    params_dict['Calmar Ratio']=calmar_ratio
    params_dict['Sharpe Ratio']=sharpe_ratio
    params_dict['Sortino Ratio']=sortino_ratio
    params_dict['Max Drawdown']=max_drawdown

    print(f"Annualized Returns : {annualized_returns}\nMax Drawdown : {max_drawdown}\nCalmar ratio : {calmar_ratio}\nSharpe ratio : {sharpe_ratio}\nSortino ratio : {sortino_ratio}\n")

    return params_dict

def get_index_investment_prices(index,test_period,initial_investment,download=False):
    start_dates={
        "14-18":"2014-01-01",
        "14-23":"2014-01-01",
        "18-23":"2018-11-01",
        "15-23":"2015-01-01",
    }
    end_dates={
        "14-18":"2018-12-31",
        "14-23":"2023-12-31",
        "15-23":"2023-12-31",
        "18-23":"2023-12-31",
    }
    data=pd.DataFrame()

    if index == "NIFTYBEES.NS":
        if download:
            nifty_file=yf.download(index,start=start_dates[test_period],end=end_dates[test_period])
        else:
            nifty_file=pd.read_csv('data/NIFTYBEES.csv').set_index('Date')
            nifty_file=nifty_file.loc[start_dates[test_period]:end_dates[test_period]]
        nifty_file=nifty_file.rename(columns={"Adj Close":index})
        data[index]=nifty_file[index]
        data.ffill(inplace=True)

    else:
        indices_data=pd.read_excel(f"data/indices-historical-Jan24.xlsx").set_index('Date')
        indices_data=indices_data.loc[start_dates[test_period]:end_dates[test_period]]
        data[index]=indices_data[index]
    #print(data)
    data['Final Investment']=data[index] * initial_investment / data[index][0]
    data.index=pd.to_datetime(data.index)
    return data

def get_current_strategy_prices(filter,test_period,initial_investment):
    start_dates={
        "14-18":"2014-01-01",
        "14-23":"2014-01-01",
        "18-23":"2018-11-01",
        "15-23":"2015-01-01",
    }
    end_dates={
        "14-18":"2018-12-31",
        "14-23":"2023-12-31",
        "15-23":"2023-12-31",
        "18-23":"2023-10-31",
    }
    data=pd.DataFrame()

    current_strategy_historical_path=f"data/backtest-data/10y/Dec2023/14-23/10y-test-14-23-{filter}.csv"
    data=pd.read_csv(current_strategy_historical_path).set_index('Date')
    data=data.rename(columns={"Final Investment":"changes"})
    data=data.loc[start_dates[test_period]:end_dates[test_period]]
    data.ffill(inplace=True)
    #print(data)
    data['Final Investment']=data['changes'] * initial_investment / data['changes'][0]
    data.index=pd.to_datetime(data.index)
    data.drop(columns=['changes'],inplace=True)
    return data

def generate_results_universe(stocks_filter:str,nifty_symbol:str,test_period,trading_day):
    """
    This code runs backtest for given filter. - "All Stocks", "Filter 1 Stocks", etc

    Args:
        nifty_symbol(str): Currently "NIFTYBEES.NS"
    """
    # with open('investments.json', 'r') as fp:
    #     main_filter_dict=json.load(fp) # to save date wise investment value
    buying_periods_investments={}
    main_filter_dict={}
    pb=[]
    nt=[]
    #print(os.getcwd())

    initial_investment = 12000

    investment = initial_investment/12

    investment_filter={} #investment in filter for each buying period

    #saves capital for each buying period
    capital_dict = {
    'all_aug': investment,
    'all_sep': investment,
    'all_oct': investment,
    'all_nov': investment,
    'all_dec': investment,
    'all_jan': investment,
    'all_feb': investment,
    'all_mar': investment,
    'all_apr': investment,
    'all_may': investment,
    'all_jun': investment,
    'all_jul': investment,
}
    #saves data backtesting data for each buying period
    backtesting_dict = {
    'all_aug': {'date': None, 'investment_value': None, 'final_investment':None, 'initial_investment':investment,'annualised_return':None},
    'all_sep': {'date': None, 'investment_value': None, 'final_investment':None, 'initial_investment':investment,'annualised_return':None},
    'all_oct': {'date': None, 'investment_value': None, 'final_investment':None, 'initial_investment':investment,'annualised_return':None},
    'all_nov': {'date': None, 'investment_value': None, 'final_investment':None, 'initial_investment':investment,'annualised_return':None},
    'all_dec': {'date': None, 'investment_value': None, 'final_investment':None, 'initial_investment':investment,'annualised_return':None},
    'all_jan': {'date': None, 'investment_value': None, 'final_investment':None, 'initial_investment':investment,'annualised_return':None},
    'all_feb': {'date': None, 'investment_value': None, 'final_investment':None, 'initial_investment':investment,'annualised_return':None},
    'all_mar': {'date': None, 'investment_value': None, 'final_investment':None, 'initial_investment':investment,'annualised_return':None},
    'all_apr': {'date': None, 'investment_value': None, 'final_investment':None, 'initial_investment':investment,'annualised_return':None},
    'all_may': {'date': None, 'investment_value': None, 'final_investment':None, 'initial_investment':investment,'annualised_return':None},
    'all_jun': {'date': None, 'investment_value': None, 'final_investment':None, 'initial_investment':investment,'annualised_return':None},
    'all_jul': {'date': None, 'investment_value': None, 'final_investment':None, 'initial_investment':investment,'annualised_return':None},
}
    
    weights_df = []
    warnings.filterwarnings('ignore')
    if test_period=="14-18":
        years=5
        result_path=f"{parent_folder}/backtest-data-{exp}/{years}y/{config.months[-1].replace(' ','')}/{test_period}/{stocks_filter}.csv"
        buying_periods=btc.buying_periods_5y_14_18
        year_fname=5
        parent_path=f"{parent_folder}/backtest-data-{exp}/{years}y/{config.months[-1].replace(' ','')}/{test_period}"
        os.makedirs(parent_path,exist_ok=True)
    elif test_period=="18-23":
        years=5
        result_path=f"{parent_folder}/backtest-data-{exp}/{years}y/{config.months[-1].replace(' ','')}/{test_period}/{stocks_filter}.csv"
        buying_periods=btc.buying_periods_5y_19_23
        year_fname=5
        parent_path=f"{parent_folder}/backtest-data-{exp}/{years}y/{config.months[-1].replace(' ','')}/{test_period}"
        os.makedirs(parent_path,exist_ok=True)
    elif test_period=="14-23":
        years=10
        year_fname=10
        result_path=f"{parent_folder}/backtest-data-{exp}/{year_fname}y/{config.months[-1].replace(' ','')}/{test_period}/{stocks_filter}.csv"
        buying_periods=btc.buying_periods_10y_14_23
        parent_path=f"{parent_folder}/backtest-data-{exp}/{year_fname}y/{config.months[-1].replace(' ','')}/{test_period}"
        os.makedirs(parent_path,exist_ok=True)


    for month_name, month_list in tqdm(buying_periods.items(),unit="Buying Periods"):

        date_list = [] #list of dates for a buying period
        investment_list= [] #investment list of a buying period

        main_month_list={} #experiment - saving the investment values for each buying period
        all_month={}
        
        investments_df=pd.DataFrame()
        for month in month_list:
            monthly_datewise_dict={}
                
            #print(month)

            month_short_name, year = month.split(' ')
            file_name=month.replace(" ","")

            start_date,end_date=get_start_end_from_month(month,period=1)# get the start end date for simulating the market
            historical_data = generate_historical_data_df(start_date,end_date) #generate historical adj close prices dataframe 
    
            selected_stocks = get_selected_stocks(month,trading_day,stocks_filter)
            
            not_there=[stock for stock in selected_stocks if stock not in historical_data.columns]
            nt.extend(not_there)
            selected_stocks = [stock for stock in selected_stocks if stock in historical_data.columns] #ensure the stocks data is present in dataframe
            historical_data = historical_data[selected_stocks] #filter out the stocks that are not needed

            #handle the case for all stocks filter where stocks with some data missing are present.
            historical_data.fillna(method='bfill',inplace=True)
            historical_data.drop(columns=list(historical_data.columns[historical_data.isna().all()]),inplace=True)
            historical_data.index = pd.to_datetime(historical_data.index)


            ####Equal Allocation to each stock
            weights = {column: 1/len(historical_data.columns) for column in historical_data.columns}

            weights_df.append(weights)
                #apply the weighted price to each stock
            weights_value = {key: value * capital_dict[month_name] for key, value in weights.items()}
            cols = []

            ########### we will normalise the historical data and multiply it by investment value #######
            
            ## Add the column (w dynamic names) of each stock's investment along with their daily returns.
            monthly_dict = {}

            for column in historical_data.columns: 
                new_col = f'{column}_Portfolio'
                historical_data[column] = historical_data[column].astype(float)
                historical_data[new_col] = historical_data[column] / float(historical_data[column].iloc[0]) * weights_value[column]
                monthly_dict[column]=round(historical_data[new_col].iloc[0],2)
                cols.append(new_col)
            investment_filter[month]=monthly_dict

            #Final dataframe containing values of investment in each stock.
            final_df = historical_data[cols]

            #####Experiment
            #calculate the total investment value in the portfolio
            final_df['Investment'] = final_df.sum(axis=1)
            final_df['Investment'] = final_df['Investment'].round(2)

            temp_fdf=final_df
            temp_fdf.index = temp_fdf.index.strftime('%Y-%m-%d')
            datewisedict=temp_fdf['Investment'].to_dict() #what was the value of portfolio on that particular date
            main_month_list.update(datewisedict) # saving the dict values for a time frame of buying period. Fr ex: from jan 2014 to dec 2014 for Jan 2014 Buying
            #####Experiment
            
            #change the investment value for next buying period of the same month's series.
            capital_dict[month_name] = final_df['Investment'].iloc[-1]
            percent_return_dict[month] = (final_df['Investment'].iloc[-1]/final_df['Investment'].iloc[0]-1)*100
            backtesting_dict[month_name]['final_investment'] = final_df['Investment'].iloc[-1] #final of a buying period
            backtesting_dict[month_name]['annualised_return'] = (((capital_dict[month_name]/ investment) ** (1/years)) - 1)*100 #annualized returns
            investment_list.extend(final_df['Investment'].tolist()) # list of investment values
            date_list.extend(final_df.index.tolist()) # list of corresponding dates
                
            
        
        
        #save the date and investment value for different buying periods in a months' series.
        backtesting_dict[month_name]['date'] = date_list # list of dates
        backtesting_dict[month_name]['investment_value'] = investment_list #list of investment values

        buying_periods_investments[month_name]=main_month_list #experiment updating the existing buying periods dict
    
    #here, in backtesting_dict all the investment values for a date are there in lists.
    main_filter_dict[stocks_filter]=buying_periods_investments #dict containing buying period along with their investment values with date as key and value as investment.
    investment_track_path_final=f"{investment_track_path}/{stocks_filter}-{test_period}-investment-track.json"
    with open(investment_track_path_final, 'w') as fp:
        json.dump(main_filter_dict, fp,indent=4)

    #save the data regarding stocks bought, with investment value in each of them over the years.
    investment_track[stocks_filter]=investment_filter            
    with open(investment_track_path_final, 'w') as fp:
        json.dump(investment_track, fp,indent=4)
    

    initial_capital = initial_investment
    print("Initial Capital: ",initial_capital)
    strategy_final_capital = sum(capital_dict.values())
    print("Final Capital: ",round(strategy_final_capital,0))

    annualized_returns = round((((strategy_final_capital/ initial_capital) ** (1 /years)) - 1)*100,2) #annualized returns
    
    #TODO: Debug this
    # Create a DataFrame from the list of dictionaries
    # data_list = [{'Month': month, 'Date': date, 'Investment': investment}
    #             for month, data in backtesting_dict.items()
    #             for date, investment in zip(data['date'], data['investment_value'])]
    # #data_list_folder=f"{parent_folder}/backtest-data/{year_fname}y/{(config.months[-1]).replace(' ','')}/{test_period}"
    # save_data_to_json(data_list,file_path=f"{parent_path}/{stocks_filter}-datalist.json")
    
 
    detailed_path=f"{parent_folder}/backtest-data-{exp}/{year_fname}y/{(config.months[-1]).replace(' ','')}/{test_period}/detailed-results"
    os.makedirs(detailed_path,exist_ok=True)
    investment_path=f"{detailed_path}/{stocks_filter}.csv"
    strategy_portfolio=detailed_results_df(buying_periods_investments,investment_path,save=DETAILED_REPORT)
    
    if test_period != "14-23" and test_period != "15-23":
        numbers_path=f"{parent_path}/numerical_data.json"
    else:
        numbers_path=f"{parent_path}/numerical_data.json"
    try:
        with open(numbers_path,'r') as o:
            numbers=json.load(o)
    except:
        numbers={}

    print("--------"*5,f"\n{FILTER} Performance")
    #strategy_portfolio=generate_results_df(data_list)
    strategy_portfolio.to_csv(result_path)
    filter_data=calculate_params(strategy_portfolio.copy(),initial_investment=initial_investment,years=years)      
    numbers[stocks_filter]=filter_data

    index=config.nifty_etf #+config.nse
    print("--------"*5,f"\n{index} Performance")
    index_portfolio=get_index_investment_prices(index,test_period,initial_investment)
    index_portfolio.to_csv(f"{parent_path}/{index}.csv")
    benchmark_data=calculate_params(index_portfolio.copy(),initial_investment=initial_investment,years=years)
    numbers[index]=benchmark_data
    
    # print("--------"*5,f"\nCS (F2) Performance")
    # cs_filter='Filter 2 Stocks'
    # cs_portfolio=get_current_strategy_prices(cs_filter,test_period,initial_investment)
    # benchmark_data=calculate_params(cs_portfolio.copy(),initial_investment=initial_investment,years=years)
    # numbers['CS (F2)']=benchmark_data

    # index="NIFTY MIDCAP 150"
    # print("--------"*5,f"\n{index} Performance")
    # index_portfolio=get_index_investment_prices(index,test_period,initial_investment)
    # benchmark_data=calculate_params(index_portfolio.copy(),initial_investment=initial_investment,years=years)
    # numbers[index]=benchmark_data

    # index="NIFTY SMALLCAP 250"
    # print("--------"*5,f"\n{index} Performance")
    # index_portfolio=get_index_investment_prices(index,test_period,initial_investment)
    # benchmark_data=calculate_params(index_portfolio.copy(),initial_investment=initial_investment,years=years)
    # numbers[index]=benchmark_data

    with open(f"{numbers_path}",'w') as o:
        json.dump(numbers,o,indent=4)

    

FILTERS=[
        'All Stocks',
         'Filter 1 Stocks',
         'Filter 2 Stocks',
         'Filter 3 Stocks Max Sharpe 1Y',
        ]

if __name__ == "__main__":
    count=0
    trading_day="1"#config.trading_days_by_month[0]
    test_periods=["14-23","18-23","14-18"]
    

    #FILTER=FILTERS[13]#'Filter 3 Stocks'
    for FILTER in FILTERS[1:2]:
        print("\n--------- "+FILTER+" ---------")
        
        for test_period in test_periods[0:1]:
            backtest_meta_data = generate_results_universe(stocks_filter=FILTER,
                                                        nifty_symbol=nifty_symbol,
                                                        test_period=test_period,
                                                        trading_day=trading_day)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        print("--------- "+FILTER+" ---------")

    print("Done")

    
        
    


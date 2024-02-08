#custom modules
from period_functions import *
from shm_functions import *
import config
from config import parent_folder

#standard libraries
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
import time
import warnings


days=config.days
reg=config.reg
close_type=config.close_type

STOCKMEM=set()

historical_path_stocks=config.stocks_historical_data_src #path of stocks historical prices dataframe
df_s_og=pd.read_csv(historical_path_stocks).set_index('Date')

historical_path_indices=config.indices_historical_data_src #path of indices prices dataframe
df_i_og=pd.read_csv(historical_path_indices).set_index('Date')


with open(config.custom_tmi_path,'r')as t:
    custom_tmi=json.load(t)

def get_trading_stocks(quarter): # will load the stocks in universe at a give quarter.
    list_stocks=custom_tmi[quarter]
    return list_stocks

Q_M={
    "Q1":"Jan",
    "Q2":"Apr",
    "Q3":"Jul",
    "Q4":"Oct"
}

M_Q={ # complete the list
    "Jan":"Q1",
    "Feb":"Q1",
    "Mar":"Q1",
    "Apr":"Q2",
    "May":"Q2",
    "Jun":"Q2",
    "Jul": "Q3",
    "Aug": "Q3",
    "Sep": "Q3",
    "Oct": "Q4",
    "Nov": "Q4",
    "Dec": "Q4"
}

def generate_dataframe_regression1(start_date,end_date,indexes,stocks,df_s,df_i):
    
    """
    This function generates a dataframe for the given time range, stocks and index.
    """
    stocks_not_from5y=[]

    df = pd.DataFrame()
    
    cols = df_i.columns.to_list()
    # Data processing for stocks dataframe.
    miss_stocks=[stock for stock in stocks if stock not in df_s.columns]
    m_stocks=[]
    if len(miss_stocks) !=0:
        m_stocks.extend(miss_stocks)
        print(f"Missing stocks - {len(m_stocks)}")
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
            elif (NaN_ratio < 0.08 ) and not first_isna:
                final_stocks.append(stock) #case where data is missing for companies in between time range. 
            else:
                stocks_not_from5y.append(stock)


     
    #print(f"{len(final_stocks)} chosen out of {len(stocks)}")
    unclean_index=[]
    final_indexes=[]

    try:
        df_i.index = pd.to_datetime(df_i.index, format='%Y-%m-%d').date
        rows_to_drop = pd.Series(df_i.isna().mean(axis=1) >= 0.6, index=df_i.index)
        df_i = df_i[~rows_to_drop]
        df_i.ffill(inplace=True)
        f_df_i=df_i.loc[start_date:end_date]
        columns_with_nan = f_df_i.columns[f_df_i.isna().any()].tolist()
        #print(f"Missing indexes - {len(columns_with_nan)}")
        for index in indexes:
            idf=f_df_i[[index]]
            idf=idf.loc[:, ~idf.columns.duplicated()]

            try:
                nan_count = idf.isna().sum()
                NaN_ratio=nan_count/len(f_df_i[[index]])
                #choose the stocks whose whole 5 year data is available
                if (NaN_ratio == 0): #dont choose stocks which have np.NaN value
                    #print(f"{stock} in existence from 5 years")             
                    final_indexes.append(index)
                elif (NaN_ratio < 0.08 ) and not first_isna:
                    final_indexes.append(index)
                else:
                    unclean_index.append(index)
            except Exception as b:
                nan_count = idf.isna().sum().iloc[0]
                first_isna=pd.isna(idf[index].iloc[0])
                NaN_ratio=nan_count/len(f_df_i[[index]])

               # Create a mask for NaN values
                nan_mask = f_df_i[index].isna()

                # Use the groupby and cumsum to create groups of consecutive NaN values
                groups = nan_mask.ne(nan_mask.shift()).cumsum()

                # Use groupby on the created groups and count NaN values
                consecutive_nan_counts = f_df_i.groupby(groups)[index].transform('size') * nan_mask

                # Find the maximum consecutive NaN count
                max_consecutive_nan = consecutive_nan_counts.max()


                if (NaN_ratio == 0): #dont choose stocks which have np.NaN value
                    #print(f"{stock} in existence from 5 years")             
                    final_indexes.append(index)
                elif (NaN_ratio < 0.08 ) and not first_isna:
                    if max_consecutive_nan < 3:
                    # Create a mask for NaN values
                        final_indexes.append(index)
                    else:
                        #print(f"Consec. NaN : {max_consecutive_nan} for {index}")
                        unclean_index.append(index) #case where data is missing for companies in between time range. 
                else:
                    unclean_index.append(index)

        
    except Exception as e:
        print(e)
    
    print(f"Missing indexes - {len(unclean_index)}")

    #final_indexes=[index for index in indexes if index not in columns_with_nan]
    df = f_df_s.copy()
    try:
        df[final_indexes]=f_df_i[final_indexes]
    except:
        try:
            final_indexes=final_indexes.upper()
            df[final_indexes]=f_df_i[final_indexes]
        except:
            return pd.DataFrame()
    

    final_stocks_=list(set(final_stocks))
    final_df = df.dropna(subset="TTW1DOWA") #drop the dates which have NaN values acc to index since data is from NSE
    # final_df = df.dropna(axis=0, how='all')
    warnings.filterwarnings('ignore')

    final_df.ffill(inplace=True) ## Forward fill for stocks incase data is there for index but not for stock
    
    # Identify and drop duplicate columns
    final_df = final_df.loc[:, ~final_df.columns.duplicated(keep='first')]
    return final_df,final_indexes,final_stocks_

def universe_1_std(data_list,new_data_json,shm_results_path_dict,shm_excel,u1_path):
    """The function generates universe 1 excel file for standard procedure i.e stocks with +ve ShM for all 5 years.

    Args:
        data_list (dict): indices and the list of +ve ShM stocks

    Returns:
        DataFrame: Universe 1 Dataframe
    """
    reg=config.reg
    days=config.days
    close_type=config.close_type

    sheet1 = pd.DataFrame()
    index_count = 0
    unique_stocks = set()

     # compute the number of stocks having +ve ShM in each index
    for dictionary in data_list:
        
        for key, values in dictionary.items():
            curr = {}
            curr["Index Name"] = key
            count = 1
            for value in values:
                unique_stocks.add(value)
                key = f"Stock_{count}"
                curr[key] = value
                count+=1
            new_row = pd.DataFrame(curr,index=[index_count])
            index_count+=1
            sheet1 = pd.concat([sheet1,new_row])
    
    sheet1.index = sheet1["Index Name"]
    sheet1.drop("Index Name",axis=1,inplace=True)
    sheet2 = pd.DataFrame(index=list(unique_stocks))
    sheet2["Index Present"] = np.NaN
    sheet2["Index Satisfied"] = np.NaN
    sheet2.index.name = "Stock" 

    # With stock being the index, add the indexes they are present in and indexes they have satisfied
    for stock in sheet2.index:
        present = []
        satisfied = []
        for index in new_data_json[next(iter(new_data_json))]:
            temp_idx_dict= new_data_json[next(iter(new_data_json))][index]
            index_name = temp_idx_dict["index name"]
            stocks_list = temp_idx_dict['stock list']
            #add '.NS' to stocks that don't have .NS
            result_list= [item + '.NS' if not item.endswith('.NS') else item for item in stocks_list]
            #result_list = [item + ".NS" for item in stocks_list] 
            
            if stock in result_list:
                present.append(index_name)

        stock_columns = sheet1.columns
        
        for index, row in sheet1.iterrows():
            for column in stock_columns:
                if row[column] == stock:
                    satisfied.append(index)

        sheet2.loc[sheet2.index == stock,"Index Present"] = str(present)
        sheet2.loc[sheet2.index == stock,"Index Satisfied"] = str(satisfied)
    
    #Sheet 1 - Index and their +ve ShM stocks.
    #Sheet 2 - Stocks and their indexes (present and satisfying)
    writer = pd.ExcelWriter(shm_excel, engine='xlsxwriter')
    sheet1.to_excel(writer, sheet_name='Sheet-1(All Indices)')
    sheet2.to_excel(writer, sheet_name='Sheet-2(All Stocks)')
    writer.save()  
    writer.close()

    universe1 = pd.read_excel(shm_excel,sheet_name='Sheet-2(All Stocks)')

    #calculate the Min, Max, Avg ShM values for a stock for each index it is present in
    for row in universe1.iterrows(): #choose a stock
        stock_ticker = row[1]['Stock']
        index_present = eval(row[1]['Index Present']) #list of indexes the stock is present in
        shm_values = []
        shm_values = []

        for index in index_present: 
            index_symbol=get_index_symbol(index).upper()##re-open all files using symbol names
            if index_symbol ==None:
                continue
            shm_results_paths=shm_results_path_dict[index_symbol.upper()]
            try:
                results_1year_shm = pd.read_csv(shm_results_paths[0])
                shm_1year = results_1year_shm.loc[results_1year_shm["Name of the Stock"] == stock_ticker, f"ShM{days}"].tolist()[0]
                shm_values.append(shm_1year)
            except (FileNotFoundError, IndexError):
                shm_values.append(None)

            try:
                results_2year_shm = pd.read_csv(shm_results_paths[1])
                shm_2year = results_2year_shm.loc[results_2year_shm["Name of the Stock"] == stock_ticker, f"ShM{days}"].tolist()[0]
                shm_values.append(shm_2year)
            except (FileNotFoundError, IndexError):
                shm_values.append(None)

            try:
                results_3year_shm = pd.read_csv(shm_results_paths[2])
                shm_3year = results_3year_shm.loc[results_3year_shm["Name of the Stock"] == stock_ticker, f"ShM{days}"].tolist()[0]
                shm_values.append(shm_3year)
            except (FileNotFoundError, IndexError):
                shm_values.append(None)

            try:
                results_4year_shm = pd.read_csv(shm_results_paths[3])
                shm_4year = results_4year_shm.loc[results_4year_shm["Name of the Stock"] == stock_ticker, f"ShM{days}"].tolist()[0]
                shm_values.append(shm_4year)
            except (FileNotFoundError, IndexError):
                shm_values.append(None)

            try:
                results_5year_shm = pd.read_csv(shm_results_paths[4])
                shm_5year = results_5year_shm.loc[results_5year_shm["Name of the Stock"] == stock_ticker, f"ShM{days}"].tolist()[0]
                shm_values.append(shm_5year)
            except (FileNotFoundError, IndexError):
                shm_values.append(None)
        final_shm_values = [value for value in shm_values if value is not None]
               
        universe1.loc[universe1["Stock"] == stock_ticker,"Min(ShM)"] = round(min(final_shm_values) * math.sqrt(24),2)
        universe1.loc[universe1["Stock"] == stock_ticker,"Max(ShM)"] = round(max(final_shm_values) * math.sqrt(24),2)
        universe1.loc[universe1["Stock"] == stock_ticker,"Avg(ShM)"] = round((sum(final_shm_values)/len(shm_values)) * math.sqrt(24),2)
    
    
    universe1.to_csv(u1_path,index=False)
    return universe1

def generate_stock_membership_universe1(month:str,df_s,df_i,perform_regression=False):
    """The function runs regression to compute beta values and choose the stocks according to given conditions for an index.
    The function also creates the final indices-data-json for backtesting.

    Args:
        quarter (str): _description_
        method (int): _description_
        regenerate (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_



    """

    m,year=month.split(" ")
    print(month)
    quarter=f"{M_Q[m]} {year}"
    data_list = []
    q,year=quarter.split(" ")
    shm_results_path_dict={}

    #quarter_meta_data=(prev_mon,prev_year,curr_mon,curr_year)
    trading_stocks=get_trading_stocks(quarter) # function that loads the list of stocks that are there as the Base universe.
    trading_indices=list(df_i_og.columns)

    

    #beta_period=-5
    start_date,end_date=get_start_end_from_month(month,config.beta_period_5y) # this will load the number of years for which beta is calculated. Defaults to 5 yrs.
    
    #list copy
    final_stocks=trading_stocks[:]
    index_count=1

    if perform_regression == True:
        df,final_indexes, final_stocks = generate_dataframe_regression1(start_date,end_date,trading_indices,final_stocks[:],df_s.copy(),df_i.copy())
        fetched_data_columns = df.columns
        stocks_columns=[f"{stock}_10d_log_returns" for stock in final_stocks]

        ## generate log returns----->
        df = log_returns(df,fetched_data_columns) #ticker_log_returns
        
        df.index = pd.to_datetime(df.index)
        list_of_tickers = list(fetched_data_columns)

        df_1year_og = data_for_specific_time_period("1 Year",df)
        df_2year_og = data_for_specific_time_period("2 Year",df)
        df_3year_og = data_for_specific_time_period("3 Year",df)
        df_4year_og = data_for_specific_time_period("4 Year",df)
        df_5year_og = data_for_specific_time_period("5 Year",df)


        # generate rolling 10 day rolling window log returns
        list_of_tickers = list(fetched_data_columns)

        # generate rolling 10 day rolling window log returns
        df_1year_og = non_overlap_rolling_window(df_1year_og,list_of_tickers) #ticker_10d_log_returns
        df_2year_og = non_overlap_rolling_window(df_2year_og,list_of_tickers)
        df_3year_og = non_overlap_rolling_window(df_3year_og,list_of_tickers)
        df_4year_og = non_overlap_rolling_window(df_4year_og,list_of_tickers)
        df_5year_og = non_overlap_rolling_window(df_5year_og,list_of_tickers)



        #regress all stocks against each index
        for index_symbol in final_indexes[:]:
            #index_symbol=get_index_symbol(index_name).upper()
            #df = generate_dataframe_regression(start_date,end_date,index_symbol.upper(),final_stocks)
            print(f"\n********************* {index_symbol} ******************************")
            shm_results_paths=[]#list containing the paths for a given index
            shm_path=f"{parent_folder}/shm_results/{close_type}/{reg}/{str(year)}/{month}/{index_symbol.replace(' ', '')}/"
            os.makedirs(shm_path,exist_ok=True)
            
            #TODO: format df_1year in any required way
            #files that store regression data
            for i in range(1,6):
                shm_results_paths.append(f"{shm_path}{i}_year.csv")
            shm_results_path_dict[index_symbol.upper()]=shm_results_paths

            
            stocks_columns=list(set(stocks_columns))
            stocks_columns.insert(0,f"{index_symbol}_10d_log_returns")
            # adding stocks column
            df_1year = df_1year_og[stocks_columns]
            df_2year = df_2year_og[stocks_columns]
            df_3year = df_3year_og[stocks_columns]
            df_4year = df_4year_og[stocks_columns]
            df_5year = df_5year_og[stocks_columns]

            

            if not df_1year.empty:
                
                #reorder columns
                results_1year = set_default_results(days)
                results_2year = set_default_results(days)
                results_3year = set_default_results(days)
                results_4year = set_default_results(days)
                results_5year = set_default_results(days)
                
                
                

                warnings.filterwarnings('ignore')
                #pass the data frames to regression models for calculating alpha
                # Code is running in debug mode
                #pass the data frames to regression models for calculating alpha
                start_reg=time.time()


                if 'PYTHONUNBUFFERED' in os.environ:
                    # Code is running in debug mode
                    results_1year = get_results_regression(index_symbol,final_stocks,df_1year,results_1year,reg,days)
                    results_2year = get_results_regression(index_symbol,final_stocks,df_2year,results_2year,reg,days)
                    results_3year = get_results_regression(index_symbol,final_stocks,df_3year,results_3year,reg,days)
                    results_4year = get_results_regression(index_symbol,final_stocks,df_4year,results_4year,reg,days)
                    results_5year = get_results_regression(index_symbol,final_stocks,df_5year,results_5year,reg,days)
                else:
                    # Code is running in bash/terminal
                    iterator = zip([df_1year,df_2year,df_3year,df_4year,df_5year],
                                        [results_1year,results_2year,
                                        results_3year,results_4year,results_5year])
                    
                    results = joblib.Parallel(n_jobs=-1)(
                        joblib.delayed(get_results_regression)(
                            index_symbol,
                            final_stocks,
                            temp_df,
                            temp_res,
                            reg,
                            days) 
                        for temp_df, temp_res in iterator
                    )
                    results_1year = results[0]
                    results_2year = results[1]
                    results_3year = results[2]
                    results_4year = results[3]
                    results_5year = results[4]

                end_reg=time.time()
                reg_ex = end_reg - start_reg
                print(f"Regression Execution Time: {reg_ex} seconds")

                results_1year.to_csv(shm_results_paths[0],index=False)
                results_2year.to_csv(shm_results_paths[1],index=False)
                results_3year.to_csv(shm_results_paths[2],index=False)
                results_4year.to_csv(shm_results_paths[3],index=False)
                results_5year.to_csv(shm_results_paths[4],index=False)
            
                    
                index_count+=1 


                #Save the file containing Alpha, Beta and ShM values of stocks for a particular index and its stocks.
                results_1year = results_1year[results_1year[f'ShM{days}'] > 0]  
                results_2year = results_2year[results_2year[f'ShM{days}'] > 0]  
                results_3year = results_3year[results_3year[f'ShM{days}']  > 0]  
                results_4year = results_4year[results_4year[f'ShM{days}']  > 0]  
                results_5year = results_5year[results_5year[f'ShM{days}']  > 0]  

                column_1year = set(results_1year["Name of the Stock"])
                column_2year = set(results_2year["Name of the Stock"])
                column_3year = set(results_3year["Name of the Stock"])
                column_4year = set(results_4year["Name of the Stock"])
                column_5year = set(results_5year["Name of the Stock"])

                strong_stocks = column_1year.intersection(column_2year, column_3year,column_4year, column_5year)
                print(f"Strong Stocks: {len(strong_stocks)}")
                data = {  **{index_symbol: list(strong_stocks)} }
                data_list.append(data)
                data_list_path=f"{parent_folder}/universe-1/regression-data/{month}_datalist.json"
                with open(data_list_path,'w') as data_json:
                    json.dump(data_list,data_json,indent=4)
              
            else:
                print("Skipping the Index: ",index_symbol)
    else:
        data_list_path=f"{parent_folder}/universe-1/regression-data/{month}_datalist.json"
        print("Not computing Alphas & Betas values. Pass perform_regression = True, to compute.")         


def generate_specific_quarter(year,q,perform_regression=True):
    """
    The function will re-write the older membership data.

    Args:
        year (_type_): _description_
        q (_type_): _description_
        perform_regression (bool, optional): _description_. Defaults to True.
    """
    membership_path=f"{parent_folder}/stock-membership/indices-data-Q{q}-{year}.json"
    quarter=f"Q{q} {year}"
    try:
        strt=time.time()
        uni1=generate_stock_membership_universe1(quarter,df_s_og,df_i_og,perform_regression=perform_regression)
        id_count+=1
        end=time.time()
        print(f"\n Time taken for this quarter {end-strt} seconds")
    except Exception as e:
        print(f"Error: {e}")


def generate_multiple(start_year,end_year):
    not_done=[]
    id_count=0

    all_time=time.time()
    for year in range(start_year,end_year+1):
        for d in range(1,5):
            quarter=f"Q{d} {year}"
            if quarter=="Q1 2014":
                continue
            membership_path=f"{parent_folder}/stock-membership/indices-data-Q{d}-{year}.json"
            if not os.path.exists(membership_path):
                try:
                    strt=time.time()
                    uni1=generate_stock_membership_universe1(quarter,df_s_og,df_i_og,perform_regression=True)
                    id_count+=1
                    end=time.time()
                    print(f"\n Time taken for this quarter {end-strt} seconds")
                except Exception as e:
                    not_done.append(f"Q{d} {year}")
                    continue
            else:
                print(f" Already there Q{d} {year}")
            break
        break
    print(f"Not done: {not_done}")

    all_end_time=time.time()
    print(f" All the indices done in {(all_end_time-all_time)/60} minutes. Total quarters = {id_count}")




if __name__ == "__main__":
    for month in config.months[80:81]:#[68:69]:
        generate_stock_membership_universe1(month,df_s_og,df_i_og,perform_regression=config.perform_regression)
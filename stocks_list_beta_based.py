import pandas as pd
import config
import json
import os




# rel=""
parent_folder=f"./shm_results"

# Indices data Historical
historical_path_indices = config.indices_historical_data_src
df_i_og = pd.read_csv(historical_path_indices).set_index('Date')

# List of indexes for which membership is to be checked.
index_symbols= config.index_symbols # df_i_og.columns.to_list()

def fetch_alpha_beta_files(month,close_type=config.close_type,reg=config.reg,parent_folder=parent_folder):
    m,year=month.split(" ")
    shm_results_path_dict={}
    for index_symbol in index_symbols[:]:
        shm_results_paths=[]#list containing the paths for a given index
        shm_path=f"{parent_folder}/{close_type}/{reg}/{str(year)}/{month}/{index_symbol.replace(' ', '')}/"
        for i in range(1,6):
            shm_results_paths.append(f"{shm_path}{i}_year.csv")
        shm_results_path_dict[index_symbol.upper()]=shm_results_paths
    return shm_results_path_dict


def generate_list_of_stocks(beta_value,month):
    stocks_throughout_indices=[]
    ab_files=fetch_alpha_beta_files(month)
    for index_symbol in index_symbols[:]:
        index_data_paths=ab_files[index_symbol]
        try:
            year_1=pd.read_csv(index_data_paths[0])
            year_2=pd.read_csv(index_data_paths[1])
            year_3=pd.read_csv(index_data_paths[2])
            year_4=pd.read_csv(index_data_paths[3])
            year_5=pd.read_csv(index_data_paths[4])
            
            # Logic to Filter out Stocks
            year_1_stocks=year_1[(year_1['β10'] > beta_value) & (year_1['ShM10'] > 0)]['Name of the Stock'].tolist()
            year_2_stocks=year_1[(year_2['β10'] > beta_value) & (year_2['ShM10'] > 0)]['Name of the Stock'].tolist()
            year_3_stocks=year_1[(year_3['β10'] > beta_value) & (year_3['ShM10'] > 0)]['Name of the Stock'].tolist()
            year_4_stocks=year_1[(year_4['β10'] > beta_value) & (year_4['ShM10'] > 0)]['Name of the Stock'].tolist()
            year_5_stocks=year_1[(year_5['β10'] > beta_value) & (year_5['ShM10'] > 0)]['Name of the Stock'].tolist()

            #logic to save the stocks
            strong_stocks=set(year_1_stocks).intersection(
                                                            set(year_2_stocks),
                                                            set(year_3_stocks),
                                                            set(year_4_stocks),
                                                            set(year_5_stocks)
                                                            )
            stocks_throughout_indices.extend(list(strong_stocks))
        except Exception as e:
            print(f"Skipping {index_symbol}, : {beta_value}")
    return list(set(stocks_throughout_indices))



def generate_custom_beta_membership(beta,month,parent_path):
    days_dict= config.trading_days_by_month[month]
    # trading_day=config.trading_days[0]
    date=days_dict["1"]
    os.makedirs(parent_path,exist_ok=True)
    stocks_list_json=f"{parent_path}/{month}.json"
    with open(stocks_list_json,"w") as f:
        dict_tmp={}
        date_dict={}
        u1_stocks_list=generate_list_of_stocks(beta,month)
        date_dict['Filter 1 Stocks']=u1_stocks_list
        dict_tmp[date]=date_dict
        json.dump(dict_tmp,f,indent=3)



if __name__=="__main__":

    for month in config.months[114:115]:
        # generate_custom_beta_membership(beta=0.5,month=month,parent_path=f"{parent_folder}/stocks-list-5")
        # generate_custom_beta_membership(beta=0.6,month=month,parent_path=f"{parent_folder}/stocks-list-6")
        # generate_custom_beta_membership(beta=0.65,month=month,parent_path=f"{parent_folder}/stocks-list-65")
        generate_custom_beta_membership(beta=0.7,month=month,parent_path=f"{parent_folder}/stocks-list-7")
        
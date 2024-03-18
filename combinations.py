import json
import random
import config
import backtest_config as btconfig

months = []
for key, item in btconfig.buying_periods_10y_14_23.items():
    months += item
    

    # Create permutations --->
def saving_permutations_to_json(permutation_list:list,file_path:str):
    # Specify the file path
    print(f"Saving to ... {file_path}")
    # os.makedirs(os.path.dirname(file_path), exist_ok=True) # creat directory if it doesnt exist
    # Save the list to a JSON file
    with open(file_path, "w") as json_file:
        json.dump(permutation_list, json_file)

def generate_stock_permutations(stock_list, num_permutations, num_companies):
    permutations = []

    # Generate permutations one by one until reaching the desired number
    i = 0
    while i < num_permutations:
        random.shuffle(stock_list)
        permutation = stock_list[:num_companies]

        # Check if the permutation is already in the list
        if permutation not in permutations:
            permutations.append(permutation)
            i += 1
    
    return permutations

# def create_permutations_for_backtest(data,num_permutations, uni, num_companies, months=months):
#     permutations = []
#     for month in months:
#         stocks = data[config.trading_days_by_month["1"]]["Filter 1 Stocks"]
#         print(stocks)
#         path_to_permutation_json = f"./permutations/permutations_{month}_{uni}.json"
#         permutations = generate_stock_permutations(stocks, num_permutations=num_permutations, num_companies=num_companies)
#         saving_permutations_to_json(permutations,path_to_permutation_json)
        
if __name__ == "__main__":
    for month in months:
        # month = "Jul 2023"
        data_location = f"./shm_results/stocks-list-65/{month}.json"
        with open(data_location, "r") as f:
            data = json.load(f)
            
        stocks = data[config.trading_days_by_month[month]["1"]]["Filter 2 Stocks etf"]
        path_to_permutation_json = f"./permutations-65/permutations_{month}_u2etf.json"
        permutations = generate_stock_permutations(stocks, num_permutations=500, num_companies=8)
        saving_permutations_to_json(permutations,path_to_permutation_json)
    # create_permutations_for_backtest(data, uni = "u1", num_permutations = 500, num_companies = 8)
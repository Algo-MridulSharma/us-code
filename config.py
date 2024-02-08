import json
import os
parent_folder="./"

shm_results_path="" # alpha beta values path
stock_universe_path=""
custom_tmi_path="./stock_data/quater(2013-23).json"
perform_regression=True

stocks_historical_data_src="./stock_data/US_Market_totalClose.csv"
indices_historical_data_src="./indices_data/Index_df.csv"

days = 10   # window period for n days of return
close_type="Adj Close" #close price type
window_size=10 #log return days
num_stocks=8 #stocks to be chosen at filter 3
reg="LR" #type of regression
beta_period_1y=-1 #period for which beta should be calculated
beta_period_5y=-5 #period for which beta should be calculated
beta_value=0.5

#dict mapping months with the quarter months
MAPPING_DICT={
    "Jan": "Jan",
    "Feb": "Jan",
    "Mar": "Jan",
    "Apr": "Apr",
    "May": "Apr",
    "Jun": "Apr",
    "Jul": "Jul",
    "Aug": "Jul",
    "Sep": "Jul",
    "Oct": "Oct",
    "Nov": "Oct",
    "Dec": "Oct"
}

monthsdict={
        "Apr":"Q2",
        "Jan":"Q1",
        "Jul":"Q3",
        "Oct":"Q4"
    }
'''
try:
    with open('meta_data/nifty_symbols.json') as f:
        nifty_symbols = json.load(f)
except Exception as e:
    with open('/meta_data/nifty_symbols.json') as f:
        nifty_symbols = json.load(f)
'''
months=('Jan 2013', 'Feb 2013', 'Mar 2013', 'Apr 2013', 'May 2013', 'Jun 2013', 'Jul 2013', 'Aug 2013', 'Sep 2013', 'Oct 2013', 'Nov 2013', 'Dec 2013',
         'Jan 2014', 'Feb 2014', 'Mar 2014', 'Apr 2014', 'May 2014', 'Jun 2014', 'Jul 2014', 'Aug 2014', 'Sep 2014', 'Oct 2014', 'Nov 2014', 'Dec 2014',
         'Jan 2015', 'Feb 2015', 'Mar 2015', 'Apr 2015', 'May 2015', 'Jun 2015', 'Jul 2015', 'Aug 2015', 'Sep 2015', 'Oct 2015', 'Nov 2015', 'Dec 2015',
         'Jan 2016', 'Feb 2016', 'Mar 2016', 'Apr 2016', 'May 2016', 'Jun 2016', 'Jul 2016', 'Aug 2016', 'Sep 2016', 'Oct 2016', 'Nov 2016', 'Dec 2016',
         'Jan 2017', 'Feb 2017', 'Mar 2017', 'Apr 2017', 'May 2017', 'Jun 2017', 'Jul 2017', 'Aug 2017', 'Sep 2017', 'Oct 2017', 'Nov 2017', 'Dec 2017',
         'Jan 2018', 'Feb 2018', 'Mar 2018', 'Apr 2018', 'May 2018', 'Jun 2018', 'Jul 2018', 'Aug 2018', 'Sep 2018', 'Oct 2018', 'Nov 2018', 'Dec 2018',
         'Jan 2019', 'Feb 2019', 'Mar 2019', 'Apr 2019', 'May 2019', 'Jun 2019', 'Jul 2019', 'Aug 2019', 'Sep 2019', 'Oct 2019', 'Nov 2019', 'Dec 2019',
         'Jan 2020', 'Feb 2020', 'Mar 2020', 'Apr 2020', 'May 2020', 'Jun 2020', 'Jul 2020', 'Aug 2020', 'Sep 2020', 'Oct 2020', 'Nov 2020', 'Dec 2020',
         'Jan 2021', 'Feb 2021', 'Mar 2021', 'Apr 2021', 'May 2021', 'Jun 2021', 'Jul 2021', 'Aug 2021', 'Sep 2021', 'Oct 2021', 'Nov 2021', 'Dec 2021',
         'Jan 2022', 'Feb 2022', 'Mar 2022', 'Apr 2022', 'May 2022', 'Jun 2022', 'Jul 2022', 'Aug 2022', 'Sep 2022', 'Oct 2022', 'Nov 2022', 'Dec 2022',
         'Jan 2023', 'Feb 2023', 'Mar 2023', 'Apr 2023', 'May 2023', 'Jun 2023', 'Jul 2023', 'Aug 2023', 'Sep 2023', 'Oct 2023', 'Nov 2023', 'Dec 2023',)# complete this tuple
print(months[72])
print(len(months))
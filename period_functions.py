import datetime as dt
from datetime import datetime


def get_start_end_from_date(date,period:float):
    """ 
    This function generates the Start-End dates for a given period based on period(in yrs).
    For input (datetime.date(2020,1,1),-1)
    Output:
    (datetime.date(2019, 1, 1), datetime.date(2019, 12, 31))
    period values( +/- ):
    0.25 - 3 months
    0.5 - 6 months
    1 - 1 year
    3 -  3 year
    5 - 5 year


    Args:
        date (DateTime Object): Date with whose respect we need the time periods
        period (float): number of years from date. Ex: -ve 1 for last 1 years'date and +1 for next 1 years'

    Returns:
        Tuple: Start Date, End Date
    """
    if isinstance(date, str):
        date = dt.datetime.strptime(date, '%Y-%m-%d').date()
        if period < 0:
            start_date = date + dt.timedelta(days=period*365)
            end_date = date - dt.timedelta(days=1)
        elif period > 0:
            start_date = date
            end_date = date + dt.timedelta(days=period*365.25)
        return start_date, end_date
    else:
        #date = dt.datetime.strptime(date, '%Y-%m-%d').date()
        if period < 0:
            start_date = date + dt.timedelta(days=period*365)
            end_date = date - dt.timedelta(days=1)
        elif period > 0:
            start_date = date
            end_date = date + dt.timedelta(days=period*365.25)
        return start_date.date(), end_date.date()

def get_start_end_from_month(month,period:int):
    """ The function returns start-end dates for a given period.
    The period must be full year,

    For input ("Jan 2020",-1)
    Output:
    (datetime.date(2019, 1, 1), datetime.date(2019, 12, 31))
    period values( +/- ), float:
    1 - 1 year
    3 -  3 year
    5 - 5 year


    Args:
        month (str): Example: "Mar 2020"
        period (int): number of years

    Returns:
        Tuple: 'datetime.date' objects in a tuple
    """
    
    
    mon, year = month.split(' ')
    if period<0:
        months_dict = {
        'Jan': ['1 January ' + str((int(year)+period)), '31 December ' + str(int(year)-1)],
        'Feb': ['1 February ' + str((int(year)+period)), '31 January ' + str(year)],
        'Mar': ['1 March ' + str((int(year)+period)), '29 February ' + str(year)],
        'Apr': ['1 April ' + str((int(year)+period)), '31 March ' + str(year)],
        'May': ['1 May ' + str((int(year)+period)), '30 April ' + str(year)],
        'Jun': ['1 June ' + str((int(year)+period)), '31 May ' + str(year)],
        'Jul': ['1 July ' + str((int(year)+period)), '30 June ' + str(year)],
        'Aug': ['1 August ' + str((int(year)+period)), '31 July ' + str(year)],
        'Sep': ['1 September ' + str((int(year)+period)), '31 August ' + str(year)],
        'Oct': ['1 October ' + str((int(year)+period)), '30 September ' + str(year)],
        'Nov': ['1 November ' + str((int(year)+period)), '31 October ' + str(year)],
        'Dec': ['1 December ' + str((int(year)+period)), '30 November ' + str(year)]
        }
    if period>=0:
        months_dict = {
        'Jan': ['1 January ' + str((int(year))), '31 December ' + str(int(year))],
        'Feb': ['1 February ' + str((int(year))), '31 January ' + str(int(year)+period)],
        'Mar': ['1 March ' + str((int(year))), '29 February ' + str(int(year)+period)],
        'Apr': ['1 April ' + str((int(year))), '31 March ' + str(int(year)+period)],
        'May': ['1 May ' + str((int(year))), '30 April ' + str(int(year)+period)],
        'Jun': ['1 June ' + str((int(year))), '31 May ' + str(int(year)+period)],
        'Jul': ['1 July ' + str((int(year))), '30 June ' + str(int(year)+period)],
        'Aug': ['1 August ' + str((int(year))), '31 July ' + str(int(year)+period)],
        'Sep': ['1 September ' + str((int(year))), '31 August ' + str(int(year)+period)],
        'Oct': ['1 October ' + str((int(year))), '30 September ' + str(int(year)+period)],
        'Nov': ['1 November ' + str((int(year))), '31 October ' + str(int(year)+period)],
        'Dec': ['1 December ' + str((int(year))), '30 November ' + str(int(year)+period)]
        }
    
    # Convert the date string to a datetime object
    start_date = dt.datetime.strptime(months_dict[mon][0], '%d %B %Y').date()
    try:
        end_date = dt.datetime.strptime(months_dict[mon][1], '%d %B %Y').date()
    except:
        if period<0:
            end_date = dt.datetime.strptime('28 February ' + str(int(year)), '%d %B %Y').date()
        else:
            end_date = dt.datetime.strptime('28 February ' + str(int(year)+period), '%d %B %Y').date()
    
    return start_date, end_date

if __name__ == '__main__':
    # Test case 1: Valid input with a positive period
    
    input_month_1 = "Mar 2013"
    period_1 = -5

    print(get_start_end_from_month(input_month_1,period_1))

    print(get_start_end_from_date(dt.date(2020,1,1),-1))
    
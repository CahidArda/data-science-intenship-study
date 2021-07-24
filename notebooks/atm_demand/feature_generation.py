import numpy as np
import pandas as pd

# input:    dataframe with columns: ['CashIn', 'CashOut']
# do:       remove outliers and interpolate. inplace=True by default. Maybe add an option
def clean_data(df, freq='D', drop_zeros=True):    
    df.index = pd.to_datetime(df.index) # switch to datetimeIndex

    # clean outliers
    # This may not be the best course of action when we are looking at a single ATM
    if drop_zeros:
        df[df['CashIn'] == 0] = np.NaN

    df = df.resample('D').asfreq()  # upsample
    df = df.interpolate()           # interpolate (Maybe add better methods later)

    return df

# input:    all the atm data, an atm ID
# do:       get the data of the given atm and clean it
# return:   dataframe with ['CashIn', 'CashOut'] as columns for the given ATM ID
def get_atm(df, atm_id):
    
    atm_df = df[df['AtmId'] == atm_id].copy()
    atm_df.drop(columns = 'AtmId', inplace = True)
    atm_df.set_index('HistoryDate', inplace = True)
    atm_df.index = pd.to_datetime(atm_df.index)
    
    return atm_df

# Craete windows from a series
def get_windows(series, size, prefix = "", drop_t = False):
    frame = pd.DataFrame(series.copy())
    frame.columns = ['t']

    for shift_by in range(1, size+1):
        frame[("" if (prefix=="") else prefix+"_") + "t-" + str(shift_by)] = frame['t'].shift(shift_by)

    frame.dropna(inplace = True)

    if drop_t:
        frame.drop(columns=['t'], inplace=True)

    return frame

'''
# get a series representing the index of dates
def get_day_of_the_week_index(datetimeIndex, name="Day_Index"):
    first_date = datetimeIndex[0]
    first_date_index = int(first_date.strftime('%w'))
    sequence = np.roll(np.arange(7), 1-first_date_index)

    l = len(datetimeIndex)
    day_of_the_week_index = pd.concat([pd.Series(sequence)] * (int(l/7)+1))[:l]
    day_of_the_week_index.name = name
    day_of_the_week_index.index = datetimeIndex

    return day_of_the_week_index
'''

def format_dates(datetimeIndex, new_format, name):
    new_index = datetimeIndex.to_series().apply(lambda x: int(x.strftime(new_format)))
    new_index.name = name
    return new_index

get_day_of_the_week_index  = lambda datetimeIndex: format_dates(datetimeIndex, '%u', 'Day_of_the_Week_Index') - 1
get_day_of_the_month_index = lambda datetimeIndex: format_dates(datetimeIndex, '%d', 'Day_of_the_Month_Index') - 1
get_week_of_the_year_index = lambda datetimeIndex: format_dates(datetimeIndex, '%W', 'Week_of_the_Year_Index')

# using indexes of dates, return two series representing a one-hot feature: is_weekday/is_weekend
def get_is_weekday_weekend(day_index_series):
    is_weekday = day_index_series.copy()
    is_weekday[is_weekday < 5] = 1
    is_weekday[is_weekday > 4] = 0
    is_weekday.name = "Is_Weekday"
    is_weekend = 1 - is_weekday
    is_weekend.name = "Is_Weekend"

    return is_weekday, is_weekend

# get average of last n days from a series
def get_average_of_last(series, sizes, prefix="average"):
    sizes.sort()
    windows = get_windows(series, sizes[-1], drop_t = True)

    results = []
    for size in sizes:
        averages = windows[windows.columns[:size]].mean(axis=1)
        averages.name = prefix + '_' + str(size)
        results.append(averages)
    
    return results

from datetime import timedelta

# get distance to the closest pay day
def get_distance_to_pay_days(datetimeIndex):
    # See 1st and 15th of the current month, 1st of the second month
    set_day = lambda day: lambda date: date.replace(day=day)
    curr_month_1 = datetimeIndex.to_series(index=datetimeIndex, name='curr_month_1').apply(set_day(1))
    curr_month_15 = curr_month_1.apply(set_day(15))
    curr_month_15.name = 'curr_month_15'
    next_month_1 = curr_month_1 + pd.offsets.MonthBegin(1)
    next_month_1.name = 'next_month_1'

    results = []
    pairs = [
        ('curr_month_1_delta',  curr_month_1),
        ('curr_month_15_delta', curr_month_15),
        ('next_month_1_delta',  next_month_1)
    ]
    for name, dates in pairs:
        delta = dates - datetimeIndex
        delta = delta.apply(lambda x: x.days)
        delta.name = name
        results.append(delta)

    return results

def get_trend(series, period):
    trend = 2 * series.shift(period) - series.shift(2*period)
    trend.name = series.name + "_trend_" + str(period)
    return trend

# ------------------------------------------
# Special Days
#------------------------------------------

# inputs:
#   - datetimeIndex: datetimeIndex of the original feature set
#   - dates: dates to start the range from
#   - n: size of range
#   - name: name of the series
def get_is_dates(datetimeIndex, dates, n, name):
    is_date = pd.Series(False, index = datetimeIndex, name = "is_%s"%name)
    for arefe in dates:
        is_date[pd.date_range(start=arefe, periods=n)] = True
    return is_date

# inputs:
#   - datetimeIndex: datetimeIndex of the original feature set
#   - dates: dates to end the range at
#   - n: size of range
#   - name: name of the series
def get_dates_in_n_days(datetimeIndex, dates, n, name):
    in_n_days = pd.Series(False, index = datetimeIndex, name = "%s_in_%d_days"%(name, n))
    for arefe in dates:
        in_n_days[pd.date_range(end=arefe, periods=n)] = True
    return in_n_days

RAMAZAN_AREFES = [
        '2016-7-4',
        '2017-6-24',
        '2018-6-14',
        '2019-6-4'
    ]

KURBAN_AREFES = [
        '2016-9-11',
        '2017-8-31',
        '2018-8-20',
        '2019-8-10'
    ]

get_is_ramazan        = lambda datetimeIndex: get_is_dates(datetimeIndex, RAMAZAN_AREFES, 4, 'ramazan')
get_ramazan_in_7_days = lambda datetimeIndex: get_dates_in_n_days(datetimeIndex, RAMAZAN_AREFES, 7, 'ramazan')
get_is_kurban         = lambda datetimeIndex: get_is_dates(datetimeIndex, KURBAN_AREFES, 5, 'kurban')
get_kurban_in_7_days  = lambda datetimeIndex: get_dates_in_n_days(datetimeIndex, KURBAN_AREFES, 7, 'kurban')

def get_special_dates_index(df):
    special_dates = pd.Series(0, index = df.index, name='Special_Dates_Index')
    for i, feature in enumerate(['is_ramazan', 'ramazan_in_7_days', 'is_kurban','kurban_in_7_days']):
        special_dates[df[feature] == 1] = i + 1
    return special_dates

# ------------------------------------------
# Generate feature function
# ------------------------------------------

def get_date_features(datetimeIndex):
    will_merge = []

    day_of_the_week_index = get_day_of_the_week_index(datetimeIndex)
    will_merge.append(day_of_the_week_index)
    will_merge.append(pd.get_dummies(day_of_the_week_index, prefix="Day_Index_"))
    
    will_merge.append(get_day_of_the_month_index(datetimeIndex))
    will_merge.append(get_week_of_the_year_index(datetimeIndex))

    will_merge.extend(get_is_weekday_weekend(day_of_the_week_index))
    will_merge.extend(get_distance_to_pay_days(datetimeIndex))
    
    for f in [get_is_ramazan, get_ramazan_in_7_days, get_is_kurban, get_kurban_in_7_days]:
        will_merge.append(f(datetimeIndex))

    result = pd.concat(will_merge, axis=1)
    result = pd.concat([result, get_special_dates_index(result)], axis=1)

    return result

# input:    dataframe with columns: ['CashIn', 'CashOut'], target variables
# do:       generate a feature set for the given date
# return:   return the feature set
def get_feature_sets(df, targets):
    will_merge = [df]

    # CashIn/CashOut averages of the last week/month
    sizes = [7, 30]
    for target in targets:
        will_merge.extend(get_average_of_last(df[target], sizes, target + "_average"))

    for target in targets:
        will_merge.append(get_trend(df[target], 7))

    # Last 14 days of CashIn and CashOut
    # These windows are actually created twice at the moment. One here and one inside get_average_of_last function
    # We can update to calculate windows only once later.
    will_merge.append(get_windows(df['CashIn'], 14, 'CashIn', drop_t=True))
    will_merge.append(get_windows(df['CashOut'], 40, 'CashOut', drop_t=True))

    will_merge.append(get_date_features(df.index).astype('int8'))

    result = pd.concat(will_merge, axis=1)
    result.dropna(inplace=True)
    
    return result
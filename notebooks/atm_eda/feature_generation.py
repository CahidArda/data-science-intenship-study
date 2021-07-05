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

# get a series representing the index of dates
def get_day_indexes(datetimeIndex, name="Day_Index"):
    first_date = datetimeIndex[0]
    first_date_index = int(first_date.strftime('%w'))
    sequence = np.roll(np.arange(7), 1-first_date_index)

    l = len(datetimeIndex)
    weekdays = pd.concat([pd.Series(sequence)] * (int(l/7)+1))[:l]
    weekdays.name = name
    weekdays.index = datetimeIndex

    return weekdays

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

# get distance to the closest work day
def get_distance_to_work_days(datetimeIndex):
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

# input:    dataframe with columns: ['CashIn', 'CashOut'], target variables
# do:       generate a feature set for the given date
# return:   return the feature set
def get_feature_sets(df, targets):
    will_merge = [df]

    weekdays = get_day_indexes(df.index)
    will_merge.append(weekdays)

    week_days_one_hot = pd.get_dummies(weekdays, prefix = "Day")    
    will_merge.append(week_days_one_hot)

    # Weekday/weekend (one-hot)
    is_weekday, is_weekend = get_is_weekday_weekend(weekdays)
    will_merge.append(is_weekday)
    will_merge.append(is_weekend)

    # CashIn/CashOut averages of the last week/month
    sizes = [7, 30]
    for target in targets:
        will_merge.extend(get_average_of_last(df[target], sizes, target + "_average"))

    will_merge.extend(get_distance_to_work_days(df.index))

    for target in targets:
        will_merge.append(get_trend(df[target], 7))

    # Last 14 days of CashIn and CashOut
    # These windows are actually created twice at the moment. One here and one inside get_average_of_last function
    # We can update to calculate windows only once later.
    for target in targets:
        will_merge.append(get_windows(df[target], 14, target, drop_t=True))

    result = pd.concat(will_merge, axis=1)
    result.dropna(inplace=True)
    
    return result
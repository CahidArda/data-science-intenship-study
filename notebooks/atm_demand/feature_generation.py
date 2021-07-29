import numpy as np
import pandas as pd

def clean_data(df, freq='D', drop_zeros=True):
    """Clean Data Prior to Usage

    Clean dataframe before generating the feature set. NaN values and zeros
    (depending on the drop_zeros parameter) are dropped and then replaced
    with interpolation.

    Args:
        df (:obj:`DataFrame`): Dataset to clean
        freq (:obj:`str`, optional): Frequency to use for the DatetimeIndex of
            the cleaned dataset. Default frequency is days.
        drop_zeros (boolean,optional): Whether to drop zero values in the original
            dataset.

    Returns:
        Cleaned dataframe

    """

    df.index = pd.to_datetime(df.index) # switch to datetimeIndex

    # clean outliers
    # This may not be the best course of action when we are looking at a single ATM
    if drop_zeros:
        df[df['CashIn'] == 0] = np.NaN

    df = df.resample(freq).asfreq()  # upsample
    df = df.interpolate()           # interpolate (Maybe add better methods later)

    return df

def get_atm(df, atm_id):
    """Get Data of a Single ATM

    Get data of a single ATM from the original dataset.

    Args:
        df (:obj:`DataFrame`): DataFrame to extract ATM data from
        atm_id (int): ID of the ATM
    
    Returns:
        Dataframe with data of the given ATM

    """
    atm_df = df[df['AtmId'] == atm_id].copy()
    atm_df.drop(columns = 'AtmId', inplace = True)
    atm_df.set_index('HistoryDate', inplace = True)
    atm_df.index = pd.to_datetime(atm_df.index)
    
    return atm_df

def get_windows(series, size, prefix = "", drop_t = False):
    """Get Windows from Series

    Args:
        series (:obj:`Series`): Seris to generate windows from
        size (int): Size of the window
        prefix (:obj:`str`, optional): Prefix to generate columns names of the window
            dataset with. "" by default.
        drop_t (boolean): Whether the day t is dopped from the returned dataset.

    Returns:
        Window dataset generated from the series.

    Examples:
        >>> df['CashIn'], 14, 'CashIn', drop_t=True)

    """
    frame = pd.DataFrame(series.copy())
    frame.columns = ['t']

    for shift_by in range(1, size+1):
        frame[("" if (prefix=="") else prefix+"_") + "t-" + str(shift_by)] = frame['t'].shift(shift_by)

    frame.dropna(inplace = True)

    if drop_t:
        frame.drop(columns=['t'], inplace=True)

    return frame

def format_dates(datetimeIndex, new_format, name):
    """

    Generate series from a datetimeIndex

    Args:
        datetimeIndex (:obj:`DatetimeIndex`): datetimeIndex to generate series with
            given format from
        new_format (:obj:`str`): Format to use as strftime method parameter
        name (:obj: `str`): Name of the generated series

    Returns:
        Series generated with the given format

    Examples:
        >>> format_dates(datetimeIndex, '%u', 'Day_of_the_Week_Index')
    
    """
    new_index = datetimeIndex.to_series().apply(lambda x: int(x.strftime(new_format)))
    new_index.name = name
    return new_index

get_day_of_the_week_index   = lambda datetimeIndex: format_dates(datetimeIndex, '%u', 'Day_of_the_Week_Index') - 1
get_day_of_the_month_index  = lambda datetimeIndex: format_dates(datetimeIndex, '%d', 'Day_of_the_Month_Index') - 1
get_week_of_the_year_index  = lambda datetimeIndex: format_dates(datetimeIndex, '%W', 'Week_of_the_Year_Index')
get_month_of_the_year_index = lambda datetimeIndex: format_dates(datetimeIndex, '%m', 'Month_of_the_Year_Index') - 1

def get_is_weekday_weekend(day_index_series):
    """Get weekday/weekend Feature Series

    Args:
        day_index_series (:obj:`Series`): the day_of_the_week_index series

    Returns:
        Two series as a tuple: Is_Weekday and Is_Weekend
    """
    is_weekday = day_index_series.copy()
    is_weekday[is_weekday < 5] = 1
    is_weekday[is_weekday > 4] = 0
    is_weekday.name = "Is_Weekday"
    is_weekend = 1 - is_weekday
    is_weekend.name = "Is_Weekend"

    return is_weekday, is_weekend

def get_window_stats(series, sizes, prefix):
    """

    Create average of last k days as a feature series

    Args:
        series (:obj:`Series`): Series to generate average features from
        sizes (:obj:`list`): Number of last days to generate averages from.
        prefix (:obj:`str`, optional): Prefix to use when creating names for
            the new series

    Returns:
        List of series created by averaging last k days.

    Examples:
        Suppose we want to create averages of last 7 days and last 30 days as
        two different feature series:
        >>> get_window_stats(df["CashIn"], [7,30], "CashIn_average")
    """
    sizes.sort()
    windows = get_windows(series, sizes[-1], drop_t = True)

    results = []
    for size in sizes:
        averages      = windows[windows.columns[:size]].mean(axis=1)
        averages.name = prefix + '_average_' + str(size)
        results.append(averages)

        stds      = windows[windows.columns[:size]].std(axis=1)
        stds.name = prefix + '_std_' + str(size)
        results.append(stds)
    
    return results

from datetime import timedelta

# get distance to the closest pay day
def get_distance_to_pay_days(datetimeIndex):
    """

    Pay days in Turkey are 1st or 15th of every month. This method generates
    feature series by getting the offset to 1st and 15th of the current month,
    1st of the next month for every data.

    Args:
        datetimeIndex (:obj:`DatetimeIndex`): datetimeIndex to generate offsets
            from

    Returns:
        Three offset feature sets in a list

    """
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

def get_is_dates(datetimeIndex, dates, n, name):
    """

    Generates a boolean series where every date d in the dates
    list and the next n days of date d is True.

    Args:
        datetimeIndex (:obj:`DatetimeIndex`): datetimeIndex to generate the new
            boolean series from
        dates (:obj:`list`): List of dates to use as starting points of ranges
        n (int): Number of days to set True after date d from dates list
        name (:obj:`str`): Name of the resulting series

    Returns:
        Boolean series

    Example:
        Suppose we want to set Ramazan Holiday and the next 4 days true:
        >>> get_is_dates(datetimeIndex, ['2016-7-4', '2017-6-24'], 4, 'ramazan')
    """
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
    """

    Generates a boolean series where n number of days before every date d
    in the dates list is set to True.

    Args:
        datetimeIndex (:obj:`DatetimeIndex`): datetimeIndex to generate the new
            boolean series from
        dates (:obj:`list`): List of dates to use as ending points of ranges
        n (int): Number of days to set True before date d from dates list
        name (:obj:`str`): Name of the resulting series

    Returns:
        Boolean series

    Example:
        Suppose we want to set days before Ramazan Holiday true:
        >>> get_dates_in_n_days(datetimeIndex, ['2016-7-4', '2017-6-24'],  7, 'kurban')
    """
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
get_is_kurban         = lambda datetimeIndex: get_is_dates(datetimeIndex, KURBAN_AREFES,  5, 'kurban')
get_ramazan_in_7_days = lambda datetimeIndex: get_dates_in_n_days(datetimeIndex, RAMAZAN_AREFES, 7, 'ramazan')
get_kurban_in_7_days  = lambda datetimeIndex: get_dates_in_n_days(datetimeIndex, KURBAN_AREFES,  7, 'kurban')

def get_special_day_of_the_year(datetimeIndex, day, month, name):
    is_date = pd.Series(False, index = datetimeIndex, name = "is_%s"%name)
    is_date[(is_date.index.month==4) & (is_date.index.day==23)] = True
    return is_date

get_is_cocuk_bayrami      = lambda datetimeIndex: get_special_day_of_the_year(datetimeIndex, 23, 4,  'cocuk_bayrami')
get_is_isci_bayrami       = lambda datetimeIndex: get_special_day_of_the_year(datetimeIndex, 1,  5,  'isci_bayrami')
get_is_spor_bayrami       = lambda datetimeIndex: get_special_day_of_the_year(datetimeIndex, 19, 5,  'spor_bayrami')
get_is_zafer_bayrami      = lambda datetimeIndex: get_special_day_of_the_year(datetimeIndex, 30, 8,  'zafer_bayrami')
get_is_cumhuriyet_bayrami = lambda datetimeIndex: get_special_day_of_the_year(datetimeIndex, 29, 10, 'cumhuriyet_bayrami')

def get_special_dates_index(df, features, name):
    special_dates = pd.Series(0, index = df.index, name=name)
    for i, feature in enumerate(features):
        special_dates[df[feature] == 1] = i + 1
    return special_dates

# ------------------------------------------
# Clustering
# ------------------------------------------

def get_clustering_df(all_atms_feature_set, clustering_feature, target):
    """
    Method for generating a df for clustering, based on the Day_of_the_Week_Index
    feature.

    Args:
        all_atms_feature_set (:obj:`DataFrame`): Dataframe with feature sets
            of all atms to use in training/testing
        clustering_feature (:obj:`str`): categorical feature to create clustering
            df with. Column names of the clustering df will be unique values of
            this feature
        target (:obj:`str`): Target feature

    Returns:
        Dataframe with shape (n_unique_atms, 7) to be used in clustering
    """
    clustering_df = pd.DataFrame(columns=all_atms_feature_set[clustering_feature].unique(), dtype='float64')

    for atm_id in all_atms_feature_set['AtmId'].unique():
        atm_df = all_atms_feature_set[all_atms_feature_set['AtmId'] == atm_id]
        clustering_df.loc[atm_id] = atm_df.groupby(clustering_feature).mean()[target]

    clustering_df = clustering_df.divide(clustering_df.sum(axis=1), axis = 0)
    return clustering_df

def get_clustering(clustering_df, clustering_alg, n_clusters, random_state = 42):
    """
    Applies given clustering algorithm to the clustering_df. Generates a dictionary
    with labels of clustering algorithm.

    Args:
        clustering_df (:obj:`DataFrame`): Dataframe used for clustering
        clustering_alg (:obj:`sklearn.cluster`): clustering algorithm
        n_clusters (int): number of clusters
        random_state (int, optional): random state to use in clustering. Default
            value is 42.

    Returns:
        Dictionary with clustering_df index as keys and clustering labels as values

    """
    fitted_alg = clustering_alg(n_clusters=n_clusters, random_state=random_state).fit(clustering_df)

    return {i:label for i, label in zip(clustering_df.index, fitted_alg.labels_)}

def add_cluster_features(all_atms_feature_set, feature_cluster_pairs, target, clustering_alg):
    """
    Adds features by applying clustering to given features

    Args:
        all_atms_feature_set (:obj:`DataFrame`): Dataframe with feature sets
            of all atms to use in training/testing
        feature_cluster_pairs (:obj:`list`): List of two item tuples. First item is
            feature name and second item is the number of clusters for that feature.
        target (:obj:`str`): Feature to use as target when clustering.
        clustering_alg (:obj:`sklearn.cluster`): clustering algorithm
    """
    all_atms_feature_set = all_atms_feature_set.copy()

    for feature, n_clusters in feature_cluster_pairs:
        clustering_df = get_clustering_df(all_atms_feature_set, feature, target)
        d = get_clustering(clustering_df, clustering_alg, n_clusters)

        all_atms_feature_set[feature + '_ClusterId'] = all_atms_feature_set['AtmId'].map(d)
    
    return all_atms_feature_set

# ------------------------------------------
# Generate feature function
# ------------------------------------------

def get_date_features(datetimeIndex):
    will_merge = []

    day_of_the_week_index = get_day_of_the_week_index(datetimeIndex)
    will_merge.append(day_of_the_week_index)
    will_merge.append(pd.get_dummies(day_of_the_week_index, prefix="Day_Index"))
    
    will_merge.append(get_day_of_the_month_index(datetimeIndex))
    will_merge.append(get_week_of_the_year_index(datetimeIndex))
    will_merge.append(get_month_of_the_year_index(datetimeIndex))

    will_merge.extend(get_is_weekday_weekend(day_of_the_week_index))
    will_merge.extend(get_distance_to_pay_days(datetimeIndex))
    
    for f in [get_is_ramazan, get_ramazan_in_7_days,
              get_is_kurban, get_kurban_in_7_days, 
              get_is_cocuk_bayrami, get_is_isci_bayrami,
              get_is_spor_bayrami, get_is_zafer_bayrami,
              get_is_cumhuriyet_bayrami              
              ]:
        will_merge.append(f(datetimeIndex))

    result = pd.concat(will_merge, axis=1)
    result = pd.concat([result, get_special_dates_index(result, ['is_ramazan', 'ramazan_in_7_days', 'is_kurban','kurban_in_7_days'], 'Special_Lunar_Dates_Index')], axis=1)

    return result

# input:    dataframe with columns: ['CashIn', 'CashOut'], target variables
# do:       generate a feature set for the given date
# return:   return the feature set
def get_feature_sets(df, targets):
    will_merge = [df]

    # CashIn/CashOut averages of the last week/month
    sizes = [7, 14, 30]
    for target in targets:
        will_merge.extend(get_window_stats(df[target], sizes, target))

    for target in targets:
        will_merge.append(get_trend(df[target], 7))

    # Last 14 days of CashIn and CashOut
    # These windows are actually created twice at the moment. One here and one inside get_window_stats function
    # We can update to calculate windows only once later.
    will_merge.append(get_windows(df['CashIn'], 14, 'CashIn', drop_t=True))
    will_merge.append(get_windows(df['CashOut'], 28, 'CashOut', drop_t=True))

    will_merge.append(get_date_features(df.index).astype('int8'))

    result = pd.concat(will_merge, axis=1)
    result.dropna(inplace=True)
    
    return result

def get_all_atms_feature_set(df, atm_ids=None, first_n=None):
    assert atm_ids != None or (atm_ids == None and first_n != None), "You must provide atm_ids or first_n parameter"
    
    if atm_ids == None:
        atm_ids = df['AtmId'].value_counts()[:first_n].index

    feature_sets = []

    for atm_id in atm_ids:
        atm_df = get_atm(df, atm_id)
        atm_df = atm_df[:-135]
        atm_df = clean_data(atm_df, drop_zeros=True)
            
        day_of_the_week_index = get_day_of_the_week_index(atm_df.index)

        atm_df['AtmId'] = atm_id
        feature_set = get_feature_sets(atm_df, ['CashIn', 'CashOut'])
        feature_sets.append(feature_set)

    return pd.concat(feature_sets, axis=0)
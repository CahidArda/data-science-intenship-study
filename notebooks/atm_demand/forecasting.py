import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def train_test_split(X, y, split=0.2):
    """

    Same function with train_test_split from sklearn.

    """
    cut = int(X.shape[0] * split)
    return X[:-cut], X[-cut:], y[:-cut], y[-cut:]

def mape_error(y_actual, y_pred, mean=True, use_index=True):
    """MAPE Error Method

    Calculates MAPE Error

    Args:
        y_actual (:obj:`Series`): Actual values of the target variable.
        y_pred (:obj:`Series`): Predicted values of the target variable.
        mean (boolean, optional): Sets whether the method will return the mean 
            error or the error over time. True by default.

    Returns:
        Absolute percentage error (MAPE without M) or MAPE depending on
        the "mean" parameter

    """
    result = 100 * pd.Series(((y_actual - y_pred) / y_actual))
    
    if use_index:
        result.index = y_actual.index

    result = result.abs()
    if mean:
        return result.mean()
    else:
        return result

def nmae_error(y_actual, y_pred):
    return (y_pred - y_actual).abs().sum() / y_actual.sum()

from feature_generation import get_windows

def get_shifted_errors(y, size, average=[]):
    """Method for Finding Baseline Error

    Uses past day with offset k (t-k) as prediction for day t. Do this
    for 'size' number of days and generate a dataframe

    Args:
        y (:obj:`DataFrame`): Actual values of the target variables
        size (int): Size of the window dataframe to generate using the actual
            values Each column is then used as prediction for t.
        average (List, optional): Used to generate predictions using average
            of previous days with given offsets. No average prediction is used by
            default.

    Returns:
        Dataframe with target variables in columns and potential baseline errors
        in rows.

    Examples:
        Suppose we want to find baseline errors for target variables 'CashIn'
        and 'CashOut'. We want to use each of the last 40 days as prediction.
        Also we want to use averages of the last 7 and 14 days; also the averages
        of last 7, 14 and 21 days as input. We generate this baseline error dataframe
        with the following way:
        >>> get_shifted_errors(feature_set[['CashOut', 'CashIn']], 40, average=[[7,14], [7,14,21]])

    """
    shifted_errors = pd.DataFrame(dtype='float64')
    for target in y.columns:
        windows = get_windows(y[target], size, drop_t=False)
        column_name = "%s_Error" % target
        for column in windows.columns[1:]:
            shifted_errors.loc[column, column_name] = mape_error(windows['t'], windows[column])

        for offsets in average:
            offsets = [str(offset) for offset in offsets]
            shifted_errors.loc['t-(%s)'%','.join(offsets), column_name] = mape_error(windows['t'], windows[['t-%s'%offset for offset in offsets]].mean(axis=1))
    
    return shifted_errors 

def get_error_with_freq(y_actual, y_pred, error_freq='w'):
    """Get Error Over Time
    
    Using actual and predicted values for a target variable, generate error
    over time with a given frequency.

    Args:
        y_actual (:obj:`Series`): Actual values of the target variable over time.
        y_pred (:obj:`Series`): Predicted values of the target variable over time.
        error_freq (:obj:`str`, optional): When plotting error, method uses
            the average error over a period of time. Size of the period is
            set with error_freq parameter. Default period size is weeks.

    Returns:
        Error over time in a pandas series format.

    """
    errors = mape_error(y_actual, y_pred, mean=False)
    errors.dropna(inplace=True)
    errors = errors.resample(error_freq).mean()
    return errors

# ------------------------------------------
# Parameter tuning
# ------------------------------------------

def compare_model_parameter(algorithm, X, y, parameter, values, shuffle=True):
    """Compare Model Parameter Values

    Uses the values provided as values for the given parameter when training
    the ML algorithm.

    Generates a plot showcasing the train and test error for different values.
    Also prints the hyperparameter value resulting with the best test error.

    Args:
        algorithm: ML model to train and find ideal hyperparameter for
        X (:obj:`DataFrame`): Feature Set
        y (:obj:`Series`): Target variable values
        parameter (:ob:`str`): Hyperparameter to optimize
        values (:obj:`List`): List of values to try
        shuffle(boolean, optional): Whether or not the dataset is shuffled
            when splitting the dataset for training asn testing. True by default.

    Examples:
        >>> compare_model_parameter(RandomForestRegressor, X, y, 'n_estimators', list(range(3,40)), shuffle=False)

        
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    train_error = []
    test_error = []
    best = [0, None]
    values.sort()
    for value in values:
        d = {parameter: value}
        model = algorithm(**d, random_state=5)
        model.fit(X_train, y_train)

        train_error.append(mape_error(y_train, model.predict(X_train)))
        test_error.append(mape_error(y_test, model.predict(X_test)))

        if best[1] == None or test_error[-1] < best[1]:
            best = [value, test_error[-1]]


    plt.plot(values, train_error, label='Training error')
    plt.plot(values, test_error, label='Testing error')
    plt.legend()
    plt.show()

    print("Best test error: %.3f with %d trees."%(best[1], best[0]))
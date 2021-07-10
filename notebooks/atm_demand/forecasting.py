import pandas as pd
import numpy as np

# input:    two pandas series representing the actual series and the predicted series
def mape_error(y_actual, y_pred, mean=True):
    result = 100 * np.abs((y_actual - y_pred) / y_actual)
    if mean:
        return result.mean()
    else:
        return result

from feature_generation import get_windows

# input:            pandas dataframe representing the target variables, size of the window (must be higher than the highest value in averages parameter)
# input example:    average=[[7,14], [7,14,21]] calculate the averages of t-7 and t-14, then find MAPE with this average. Same for t-7, t-14 and t-21
# do:               Calculate errors by shifting and averaging. These errors are then used as base errors
def get_shifted_errors(y, size, average=[]):
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

# input: two pandas series representing the predictions and actual values
# do:    calculate the error and average it with the given frequency
def get_error_with_freq(y_actual, y_pred, error_freq='w'):
    errors = mape_error(y_actual, y_pred, mean=False)
    errors.dropna(inplace=True)
    errors = errors.resample(error_freq).mean()
    return errors

# ------------------------------------------
# Parameter tuning
# ------------------------------------------

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# input:
#   - algorithm: ML algortihm to train and test
#   - X: feature dataframe
#   - y: target series
#   - parameter: parameter to test and draw on 2D plot
#   - values: range of values in a list
# do:   iterate over the values list and train the model with each parameter instance.
#       Test these models and plot the error
def compare_model_parameter(algorithm, X, y, parameter, values, shuffle=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=shuffle)

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
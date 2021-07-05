import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# input:    two pandas series representing the actual series and the predicted series
def mape_error(y_actual, y_pred, mean=True):
    result = 100 * ((y_actual - y_pred).abs() / y_actual)
    if mean:
        return result.mean()
    else:
        return result

# input:            pandas dataframe representing the target variables, size of the window (must be higher than the highest value in averages parameter)
# input example:    average=[[7,14], [7,14,21]] calculate the averages of t-7 and t-14, then find MAPE with this average. Same for t-7, t-14 and t-21
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

# input:    Trained model, data used to train the model, actual values
# do:       Using the model and data, draw actual/predicted and the error over time
def draw_model_output(model, X, y, error_freq='w', split_from=None):

    predictions = pd.Series(model.predict(X), index=X.index)
    weekly_errors = mape_error(y, predictions, mean=False)
    weekly_errors.dropna(inplace=True)
    weekly_errors = weekly_errors.resample(error_freq).mean()

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=y.index, y=y, name='%s Actual'%y.name, line=dict(color='rgba(255,0,0,0.6)')), secondary_y=False)
    fig.add_trace(go.Scatter(x=X.index, y=predictions, name = '%s Predicted'%y.name, line=dict(color='rgba(30,30,200,0.5)')), secondary_y=False)

    fig.add_trace(go.Scatter(x=weekly_errors.index, y=weekly_errors, name="Error", line=dict(color='rgba(34, 155, 0, 0.4)', width=4)), secondary_y=True)

    # set layout title
    fig.update_layout(title='%s Prediction and Actual Comparison'%y.name + (" (train-test split from %s)"%split_from.strftime('%d.%m.%Y') if split_from else ""))
    # set x axis titles
    fig.update_xaxes(title_text="Date")
    # set y axis titles
    fig.update_yaxes(title_text="Amount", secondary_y=False)
    fig.update_yaxes(title_text="<b>MAPE</b> Error", secondary_y=True)

    fig.show()
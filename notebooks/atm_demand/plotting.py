# ------------------------------------------
# Visualising the feature set
# ------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

def correlation(feature_set, correlation_with=None):
    if correlation_with != None:
        drop_column = "CashIn" if correlation_with == "CashOut" else "CashOut"
        feature_set = feature_set[feature_set.drop(columns=drop_column).columns]
        title = "Correlation with " + correlation_with
    else:
        title = "Correlation matrix"
    corrs = feature_set.corr()

    plt.figure(figsize=(10,10))
    plt.gca().invert_yaxis()

    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    plt.pcolor(corrs, cmap = "Greens")
    plt.yticks(np.arange(0.5, len(corrs.index), 1), corrs.index)
    plt.xticks(np.arange(0.5, len(corrs.columns), 1), corrs.columns, rotation = 'vertical')
    plt.title(title)
    plt.show()

# ------------------------------------------
# Plotting error, predictions and actuals
# ------------------------------------------

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from forecasting import get_error_with_freq

# input:    Trained model, data used to train the model, actual values
# do:       Using the model and data, draw actual/predicted and the error over time
def draw_model_error(model, X, y_actual, error_freq='w', split_from=None):

    y_pred = pd.Series(model.predict(X))
    weekly_errors = get_error_with_freq(y_actual, y_pred, error_freq)
    draw_error_over_time(y_actual, y_pred, weekly_errors, split_from)

def draw_error_over_time(y_actual, y_pred, weekly_errors, split_from=None):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=y_actual.index, y=y_actual, name='%s Actual'%y_actual.name, line=dict(color='rgba(255,0,0,0.6)')), secondary_y=False)
    fig.add_trace(go.Scatter(x=y_actual.index, y=y_pred, name = '%s Predicted'%y_actual.name, line=dict(color='rgba(30,30,200,0.5)')), secondary_y=False)

    fig.add_trace(go.Scatter(x=weekly_errors.index, y=weekly_errors, name="Error", line=dict(color='rgba(34, 155, 0, 0.4)', width=4)), secondary_y=True)

    # set layout title
    fig.update_layout(title='%s Prediction and Actual Comparison'%y_actual.name + (" (train-test split from %s)"%split_from.strftime('%d.%m.%Y') if split_from else ""))
    # set x axis titles
    fig.update_xaxes(title_text="Date")
    # set y axis titles
    fig.update_yaxes(title_text="Amount", secondary_y=False)
    fig.update_yaxes(title_text="<b>MAPE</b> Error", secondary_y=True)

    fig.show()
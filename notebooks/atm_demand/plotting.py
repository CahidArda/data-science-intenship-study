import matplotlib.pyplot as plt
import numpy as np

def correlation(feature_set, correlation_with=None):
    """Function for Drawing Correlation Matrix

    Draws a correlation matrix using a feature set stored in a
    pandas dataframe.

    Args:
        feature_set (:obj:`DataFrame`): Feature set used to draw
            the correlation matrix
        correlation_with (:obj:`str`, optional): Target variable to
            create the correlation matrix with. Only accepted values
            are "CashIn" and "CashOut". A correlation matrix with both
            "CashIn" and "CashOut" columns is drawn if the default
            parameter value is used.
    """
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

def draw_model_error(model, X, y_actual, error_freq='w', split_from=None):
    """Function for Plotting Model Error

    Uses the model provided as a parameter to create predicted values of
    the target variable. Then uses the actual values and the predicted values
    of the target variable to find error over time. Plots the error over time,
    actual values and the predicted values of the target variable on a Plotly
    line plot.

    Args:
        model: machine learning model with .predict method.
        X (:obj:`DataFrame`): Dataset used as model input.
        y_actual (:obj:`Series`): Actual values of the target variable.
        error_freq (:obj:`str`, optional): When plotting error, method uses
            the average error over a period of time. Size of the period is
            set with error_freq parameter. Default period size is weeks.
        split_from (:obj:`str`, optional): Has no effect on the plot generated.
            Adds the date when the train and test data is split from as information
            to the plot title. Default value is None, no information is added to
            the title about the split date.
            
    Examples:
        Suppose we generated X_train and X_test dataframes from dataframe
        X and used X_train to train a model called forest. We can then call
        draw_model_error in the following way to plot the actual and predicted
        values with error:
        >>> draw_model_error(forest, X, y, split_from=X_train.index[-1])

    """

    y_pred = pd.Series(model.predict(X))
    weekly_errors = get_error_with_freq(y_actual, y_pred, error_freq)
    draw_error_over_time(y_actual, y_pred, weekly_errors, split_from)

def draw_error_over_time(y_actual, y_pred, error, split_from=None):
    """Function for Plotting Error

    Uses actual and predicted values of a target variable to plot with
    error over time, actual and predicted values.

    Args:
        y_actual (:obj:`Series`): Actual values of the target variable
            over time.
        y_pred (:obj:`Series`): Predicted values of the target variable
            over time.
        error (:obj:`Series`): Error over time
        split_from (:obj:`str`, optional): Has no effect on the plot generated.
            Adds the date when the train and test data is split from as information
            to the plot title. Default value is None, no information is added to
            the title about the split date.

    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=y_actual.index, y=y_actual, name='%s Actual'%y_actual.name, line=dict(color='rgba(255,0,0,0.6)')), secondary_y=False)
    fig.add_trace(go.Scatter(x=y_actual.index, y=y_pred, name = '%s Predicted'%y_actual.name, line=dict(color='rgba(30,30,200,0.5)')), secondary_y=False)

    fig.add_trace(go.Scatter(x=error.index, y=error, name="Error", line=dict(color='rgba(34, 155, 0, 0.4)', width=4)), secondary_y=True)

    # set layout title
    fig.update_layout(title='%s Prediction and Actual Comparison'%y_actual.name + (" (train-test split from %s)"%split_from.strftime('%d.%m.%Y') if split_from else ""))
    # set x axis titles
    fig.update_xaxes(title_text="Date")
    # set y axis titles
    fig.update_yaxes(title_text="Amount", secondary_y=False)
    fig.update_yaxes(title_text="<b>MAPE</b> Error", secondary_y=True)

    fig.show()
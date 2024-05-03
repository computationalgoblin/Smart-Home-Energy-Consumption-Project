import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima
from sklearn.model_selection import train_test_split

def arima_baseline_model(train, order):
    """
    Args:
    data: time series data.
    order : order of the ARIMA model args (p,d,q).
    """
    # Fit ARIMA model
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    
    # Make predictions
    pred = model_fit.predict(start=0, end=len(train)-1)
    
    # Plot data and predictions
    plt.figure(figsize=(15,5))
    plt.plot(train, c='grey', label='total use')
    plt.plot(pred, c='green', label='ARIMA Predictions')
    plt.legend(fontsize=10)
    plt.ylabel('kW')
    plt.xlabel('Time')
    plt.show()
    
    # Calculate metrics
    mae = mean_absolute_error(train, pred)
    mape = mean_absolute_percentage_error(train, pred)
    mse = mean_squared_error(train, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(train, pred)
    
    # Create DataFrame for metrics
    metrics = pd.DataFrame({
        'MAE': [mae],
        'MAPE': [mape],
        'MSE': [mse],
        'RMSE': [rmse],
        'R^2': [r2]
    })
    
    return metrics

def arima_parameter_estimation(data):
    """
    Args:
    data: time series data.
    """
    # Plot original data
    plt.figure(figsize=(15, 5))
    plt.plot(data)
    plt.title('Original Time Series Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()

    # Perform Augmented Dickey-Fuller test for stationarity
    result = adfuller(data)
    # Plot ACF and PACF
    fig, ax = plt.subplots(2, 1, figsize=(15, 5))
    plot_acf(data, ax=ax[0], lags=40)
    plot_pacf(data, ax=ax[1], lags=40)
    plt.show()

    # Create DataFrame for statistical scores
    scores = pd.DataFrame({
        'ADF Statistic': [result[0]],
        'p-value': [result[1]],
        'Critical Values': [result[4]],
    })

    return scores

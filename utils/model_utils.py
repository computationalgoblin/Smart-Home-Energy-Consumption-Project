import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def time_series_analysis(column):

    roll_mean = column.rolling(window=12).mean()
    roll_std = column.rolling(window=12).std()

    # Step 1: Test for ADF and plot the diff
    adf_result = adfuller(column)

    plt.figure(figsize=(15, 5))
    plt.plot(column.diff().dropna(), color='gray', alpha=0.5, label='Differenced Series')
    plt.plot(roll_mean.diff().dropna(), color='blue', label='Rolling Mean')
    plt.plot(roll_std.diff().dropna(), color='green', label='Rolling Std')
    plt.legend()
    plt.title('Differenced Time Series with Rolling Mean and Standard Deviation')
    plt.xlabel('Date')
    plt.ylabel('Differenced Value')
    plt.show()

    
    # Step 2: Test for seasonality using seasonal decomposition and plot the result
    decomposition = seasonal_decompose(column, model='additive')
    fig = plt.figure()  
    fig = decomposition.plot()  
    fig.set_size_inches(15, 10)
    plt.suptitle('Seasonal Decomposition')
    plt.show()
    
    plt.figure(figsize=(15, 5))
    plt.plot(column, color='gray', alpha=0.5, label='Original')
    plt.plot(roll_mean, color='blue', label='Rolling Mean')
    plt.plot(roll_std, color='green', label='Rolling Std')
    plt.legend()
    plt.title('Rolling Mean and Standard Deviation')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()
    

    
    # Step 3: DataFrame with the results of both tests
    results = pd.DataFrame(index=['ADF Statistic', 'p-value', 'Critical Value (1%)', 'Critical Value (5%)', 'Critical Value (10%)'])
    results['ADF Test'] = [adf_result[0], adf_result[1], adf_result[4]['1%'], adf_result[4]['5%'], adf_result[4]['10%']]
    if adf_result[1] <= 0.05:
        print("The time series is stationary (reject the null hypothesis)")
    else:
        print("The time series is not stationary (cannot reject the null hypothesis)")
    return results


def acf_pacf_test(data):
    """
    Args:
    data: time series data.
    """
    # Plot ACF and PACF
    fig, ax = plt.subplots(2, 1, figsize=(15, 5))
    plot_acf(data, ax=ax[0], lags= 40)
    plot_pacf(data, ax=ax[1], lags= 40)
    plt.tight_layout()
    plt.show()


def ar_baseline_model(train, test, data, lags):
    """
    Args:
    train: train data
    test: test data
    data: time series data of the whole variable.
    lags : order of the AR model args.
    """
    from statsmodels.tsa.ar_model import AutoReg
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import numpy as np

    # Fit AR model
    model = AutoReg(train, lags=lags)
    model_fit = model.fit()
    
    # Plot data and predictions
    plt.figure(figsize=(15,5))
    plt.plot(data, c='grey', label='total use')
    plt.plot(model_fit.predict(dynamic=False), c='green', label='AR Train Model')
    plt.plot(test.index, model_fit.predict(start=len(train), end=len(train)+len(test)-1), c='blue', label='AR Predictions')
    plt.legend(fontsize=10)
    plt.ylabel('kW')
    plt.xlabel('Time')
    plt.show()
    
    # Return summary
    print(model_fit.summary())

    # Perform prediction and calculate residuals
    params = model_fit.params
    predictions = model.predict(params=params, start=test.index[0], end=test.index[-1])
    
    # Calculate evaluation metrics
    r2 = r2_score(test, predictions)
    mae = mean_absolute_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test - predictions) / test)) * 100
    
    # Create a DataFrame with evaluation metrics
    evaluation_metrics = pd.DataFrame({'R2': [r2], 'MAE': [mae], 'MAPE': [mape], 'MSE': [mse], 'RMSE': [rmse]})
    
    return model, evaluation_metrics


def arima_model(train, test, data, order):
    """
    Args:
    train: train data
    test: test data
    data: time series data of the whole variable.
    order : tuple of order for the ARMA model (p, d, q).
    """
    from statsmodels.tsa.arima.model import ARIMA
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import numpy as np

    # Fit ARIMA model
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    
    # Plot data and predictions
    plt.figure(figsize=(15,5))
    plt.plot(data, c='grey', label='Total Use')
    plt.plot(model_fit.predict(start=1, end=len(data)), c='green', label='ARIMA Train Predictions')
    plt.plot(test.index, model_fit.predict(start=len(train), end=len(train)+len(test)-1), c='blue', label='ARIMA Test Predictions')
    plt.legend(fontsize=10)
    plt.ylabel('kW')
    plt.xlabel('Time')
    plt.show()
    
    # Return summary
    print(model_fit.summary())

    # Perform prediction and calculate residuals
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
    residuals = test - predictions
    
    # Plot residuals
    plt.figure(figsize=(15,5))
    plt.plot(residuals, c='green', label='Residuals')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.legend(fontsize=10)
    plt.ylabel('Residuals')
    plt.xlabel('Time')
    plt.title('Residuals Plot')
    plt.show()

    # Calculate evaluation metrics
    r2 = r2_score(test, predictions)
    mae = mean_absolute_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test - predictions) / test)) * 100
    
    # Create a DataFrame with evaluation metrics
    evaluation_metrics = pd.DataFrame({'R2': [r2], 'MAE': [mae], 'MAPE': [mape], 'MSE': [mse], 'RMSE': [rmse]})
    
    return model, evaluation_metrics, residuals


def sarimax_model(train, test, data, order, seasonal_order):

    """
    Args:
    train: train data
    test: test data
    data: time series data of the whole variable.
    order : tuple of order for the ARMA model (p, d, q).
    seasonal_order: tuple of seasonal order for the ARMA model (P, D, Q, s).
    """

    # Fit SARIMAX model
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    
    # Plot data and predictions
    plt.figure(figsize=(15,5))
    plt.plot(data, c='grey', label='Total Use')
    plt.plot(model_fit.predict(start=1, end=len(data)), c='green', label='SARIMAX Train Predictions')
    plt.plot(test.index, model_fit.predict(start=len(train), end=len(train)+len(test)-1), c='blue', label='SARIMAX Test Predictions')
    plt.legend(fontsize=10)
    plt.ylabel('kW')
    plt.xlabel('Time')
    plt.show()
    
    # Return summary
    print(model_fit.summary())

    # Perform prediction and calculate residuals
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
    residuals = test - predictions
    
    # Plot predictions and residuals
    plt.figure(figsize=(15,5))
    plt.plot(test.index, predictions, c='grey', label='Predictions')
    plt.plot(test.index, test.values, c='blue', label='Actual')
    plt.plot(test.index, residuals, c='green', label='Residuals')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.legend(fontsize=10)
    plt.ylabel('Values')
    plt.xlabel('Time')
    plt.title('Predictions and Residuals Plot')
    plt.show()
    # Calculate evaluation metrics
    r2 = r2_score(test, predictions)
    mae = mean_absolute_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test - predictions) / test)) * 100
    
    # Create a DataFrame with evaluation metrics
    evaluation_metrics = pd.DataFrame({'R2': [r2], 'MAE': [mae], 'MAPE': [mape], 'MSE': [mse], 'RMSE': [rmse]})
    
    return model, evaluation_metrics




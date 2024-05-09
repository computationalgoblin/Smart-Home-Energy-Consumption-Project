import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima import auto_arima
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from keras.layers import Bidirectional
from scipy.stats import uniform
from changefinder import ChangeFinder
from scipy import stats
import itertools


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


def time_series_analysis_dataframe(dataframe):
    results_dict = {}
    
    for column in dataframe.columns:

        adf_result = adfuller(dataframe[column])
        is_stationary = adf_result[1] <= 0.05
        
        results_dict[column] = {
            'ADF Statistic': adf_result[0],
            'p-value': adf_result[1],
            'Critical Value (1%)': adf_result[4]['1%'],
            'Critical Value (5%)': adf_result[4]['5%'],
            'Critical Value (10%)': adf_result[4]['10%'],
            'Stationary': is_stationary
        }
        
        if is_stationary:
            print(f"The time series in column '{column}' is stationary (reject the null hypothesis)")
        else:
            print(f"The time series in column '{column}' is not stationary (cannot reject the null hypothesis)")
    
    results_df = pd.DataFrame.from_dict(results_dict, orient='index')
    return results_df


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


def prophet_model(data, train_ratio=0.8, test_ratio=0.2, y='use', changepoint_list=None, regressors=[]):
    # Make dataframe for training
    trr, ter = [int(len(data) * i) for i in [train_ratio, test_ratio]]
    train, test = data[0:trr], data[trr:]

    train_df = pd.DataFrame()
    train_df["ds"] = train.index
    train_df['y'] = train[y].values
    train_df["floor"] = 0
    train_df["cap"] = 20
    # Make dataframe for prediction
    future_df = pd.DataFrame()
    future_df['ds'] = test.index
    future_df["floor"] = 0
    future_df["cap"] = 20
    # Add regressors
    for i in regressors:
        train_df[i] = train[i].values
    # Add regressors
    for i in regressors:
        future_df[i] = test[i].values

    # Filter changepoints within training data range
    if changepoint_list is not None:
        changepoints_filtered = changepoint_list[(changepoint_list >= train.index.min()) & (changepoint_list <= train.index.max())]
    else:
        changepoints_filtered = None

    # Train model with Prophet
    prophet = Prophet(growth="flat", weekly_seasonality=True, daily_seasonality=True, changepoints=changepoints_filtered, changepoint_prior_scale=0.5, seasonality_prior_scale=15)
    # Include additional regressors into the model
    for i in regressors:
        prophet.add_regressor(i)
    prophet_fit = prophet.fit(train_df)

    # Predict the future
    predictions = prophet_fit.predict(future_df)

    # Revert the transformation
    predictions["yhat"] = predictions["yhat"]  # np.exp()
    test[y] = test[y]  # np.exp()

    # Evaluating prediction
    actual_values = test[y].values
    predicted_values = predictions['yhat'].values

    mae = mean_absolute_error(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_values, predicted_values)
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

    metrics_dict = {
        'MAE': [mae],
        'MSE': [mse],
        'RMSE': [rmse],
        'R2': [r2],
        'MAPE': [mape]
    }

    metrics_df = pd.DataFrame(metrics_dict)

    # Plot predictions
    plt.figure(figsize=(15, 5))
    plt.plot(test.index, test[y], color='grey', label='Actual')  # np.exp()
    plt.plot(predictions["ds"], predictions["yhat"], color='green', label='Predictions')
    plt.xlabel('Time')
    plt.ylabel('kW')
    plt.title('Actual vs. Predicted Values')
    plt.legend()
    plt.show()

    # Model components
    prophet.plot_components(predictions)
    plt.show()

    return prophet_fit, metrics_df


def lstm_implementation_pipeline(sequence, timesteps, train_size=0.80, lstm_units=50, activation='relu', optimizer='adam', loss='mse', epochs=200, verbose=0):
    X = [sequence[i:i+timesteps] for i in range(len(sequence)-timesteps)]
    Y = [sequence[i+timesteps] for i in range(len(sequence)-timesteps)]
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    size = int(len(sequence) * train_size)
    X_train, Y_train = X[:size], Y[:size]
    X_test, Y_test = X[size:], Y[size:]

    # Model architecture
    model = Sequential()
    model.add(LSTM(lstm_units, activation=activation, input_shape=(timesteps, 1)))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss)

    # Model training
    model.fit(X_train, Y_train, epochs=epochs, verbose=verbose)

    # Prediction
    Train_pred = model.predict(X_train, verbose=0)
    Y_pred = model.predict(X_test, verbose=0)

    Train_pred_series = pd.Series(Train_pred.flatten().tolist(), index=sequence.index[:size])
    Y_pred_series = pd.Series(Y_pred.flatten().tolist(), index=sequence.index[size+timesteps:])

    # Plot
    plt.figure(figsize=(15, 4))
    plt.plot(sequence.index[:len(Train_pred_series)], sequence.values[:len(Train_pred_series)], c='lightgrey', label='train data')
    plt.plot(Train_pred_series.index, Train_pred_series.values, c='lightgreen', label='train prediction')

    # Plot test data and model prediction on test data
    plt.plot(sequence.index[len(Train_pred_series)+timesteps:], sequence.values[len(Train_pred_series)+timesteps:], c='grey', label='test data')
    plt.plot(Y_pred_series.index, Y_pred_series.values, c='green', label='model prediction')

    # Adding focus plot for test data and model prediction on test data
    plt.figure(figsize=(15, 4))
    plt.plot(sequence.index[len(Train_pred_series)+timesteps:], sequence.values[len(Train_pred_series)+timesteps:], c='grey', label='test data')
    plt.plot(Y_pred_series.index, Y_pred_series.values, c='green', label='model prediction')

    plt.legend()
    plt.show()

    # Evaluation
    mse = mean_squared_error(Y_pred, sequence.values[size+timesteps:])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(sequence.values[size+timesteps:], Y_pred)
    mape = np.mean(np.abs(Y_pred - sequence.values[size+timesteps:]) / np.abs(sequence.values[size+timesteps:]))
    r2 = r2_score(sequence.values[size+timesteps:], Y_pred)
    print('MSE: %.5f' % mse)
    print('RMSE: %.5f' % rmse)
    print('MAE: %.3f' % mae)
    print('MAPE: %.3f' % mape)
    print('R^2 score: %.3f' % r2)


def lstm_multivariate_implementation_pipeline(data, n_past=1, n_future=1, epochs=60, verbose=0):

    # Splitting data into training and testing sets
    train_size = int(len(data) * 0.7)
    train_data, test_data = data[:train_size], data[train_size:]
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []

    # Creating sequences of past and future data for training set
    for i in range(n_past, len(train_data)-n_future+1):
        X_train.append(train_data.iloc[i-n_past:i, 0:data.shape[1]])
        Y_train.append(train_data.iloc[i+n_future-1:i+n_future, 0])
    
    # Creating sequences of past and future data for testing set
    for i in range(n_past, len(test_data)-n_future+1):
        X_test.append(test_data.iloc[i-n_past:i, 0:test_data.shape[1]])
        Y_test.append(test_data.iloc[i+n_future-1:i+n_future, 0])

    # Converting lists to numpy arrays
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)

    # Building the LSTM model
    model = Sequential([
        LSTM(25, activation='relu', return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(Y_train.shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')

    # Training the model
    model.fit(X_train, Y_train, epochs=epochs, verbose=verbose)

    # Predictions on training and testing sets
    Train_pred = model.predict(X_train, verbose=0)
    Y_pred = model.predict(X_test, verbose=0)

    # Converting predictions to Pandas Series
    Y_pred_series = pd.Series(Y_pred.flatten(), index=data['use'][train_size:-n_past].index)
    Train_pred_series = pd.Series(Train_pred.flatten(), index=data['use'][n_past:train_size].index)

    # Extracting true values for testing set
    Y_test = data['use'][train_size:-n_past]

    # Plotting training and testing predictions
    plt.figure(figsize=(15, 5))
    plt.plot(data['use'][:-n_past], c='grey', label='True Values')
    plt.plot(Y_pred_series, c='green', label='Predicted Values (Test)')
    plt.plot(Train_pred_series, c='lightgreen', label='Predicted Values (Train)')
    plt.legend()

    # Plotting only test predictions
    plt.figure(figsize=(15, 5))
    plt.plot(data['use'][train_size:-n_past], c='grey', label='True Values')
    plt.plot(Y_pred_series, c='green', label='Predicted Values')
    plt.legend()
    plt.title('Test Prediction')

    # Calculating evaluation metrics
    mse = mean_squared_error(Y_pred, Y_test)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test, Y_pred)
    mape = np.mean(np.abs(Y_pred[:,0] - Y_test.values) / np.abs(Y_test.values))
    r2 = r2_score(Y_test, Y_pred)

    # Creating DataFrame with metrics
    metrics_df = pd.DataFrame({'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R^2 Score': r2}, index=[0])

    return metrics_df   


def anomaly_detector(data, ad_r=0.01, ad_order=1, ad_smooth=10):
    # Initialize ChangeFinder object with custom parameters
    cf = ChangeFinder(r=ad_r, order=ad_order, smooth=ad_smooth)
    
    # Compute change score
    change_score = [cf.update(i) for i in data]
    
    # Calculate quartiles and upper threshold
    ad_score_q1 = np.percentile(change_score, 25) 
    ad_score_q3 = np.percentile(change_score, 75) 
    thr_upper = ad_score_q3 + (ad_score_q3 - ad_score_q1) * 3
    
    plt.figure(figsize=(12, 6))
    
    # Change score plot
    plt.subplot(2, 1, 1)
    sns.lineplot(data=change_score)
    plt.axhline(y=thr_upper, color='g', linestyle='--')
    plt.title(f"Change Score & Threshold")
    plt.xlabel('Index')
    plt.ylabel('Change Score')
    
    # Detected points plot
    plt.subplot(2, 1, 2)
    sns.lineplot(data=data)
    anom_points = [data.index[i] for i, score in enumerate(change_score) if score > thr_upper]
    plt.scatter(anom_points, data[data.index.isin(anom_points)], color='g', label="Anomalies detected")
    plt.title(f"Detected Anomalies")
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Construct DataFrame of changepoints
    changepoints = pd.DataFrame({'Date': anom_points})
    
    return changepoints
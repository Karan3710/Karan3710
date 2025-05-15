#!/usr/bin/env python
# coding: utf-8

# In[16]:


import yfinance as yf

# Download historical data
ticker = 'AAPL'
data = yf.download(ticker, start='2015-01-01', end='2024-12-31')
data = data[['Close']]  # Focus on closing prices
data.dropna(inplace=True)


# In[17]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data['Close'].plot(figsize=(12, 6))
plt.title(f'{ticker} Stock Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.grid(True)
plt.show()


# In[18]:


from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(data['Close'], order=(5,1,0))  # p, d, q
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)
print(forecast)
rmse_arima = np.sqrt(mean_squared_error(data['Close'][-30:], forecast)) # Calculate RMSE for the last 30 data points
#plot
plt.plot(data['Close'], label='Actual') # Plot the 'Close' column for the actual data
plt.plot(forecast, label='Forecast', color='red')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()


# In[19]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
# Fit SARIMA model (adjust parameters)
# Instead of passing the entire DataFrame, select the 'Close' column
model = SARIMAX(data['Close'], order=(1,1,1), seasonal_order=(1,1,0,12))
result = model.fit()

# Forecast
forecast_object = result.get_forecast(steps=30)
forecast_sarima = forecast_object.predicted_mean
forecast_ci = forecast_object.conf_int()

# Calculate RMSE for SARIMA
rmse_sarima = np.sqrt(mean_squared_error(data['Close'][-30:], forecast_sarima)) # type: ignore # Calculate RMSE for the last 30 data points


# Plot
plt.plot(data['Close'], label='Actual')  # Actual data
forecast.plot(label='Forecast', color='green')  # Forecast line
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='k', alpha=0.1)  # Confidence interval
plt.title('SARIMA Forecast')
plt.legend()
plt.show()

# 📏 SARIMA Forecasting
model_sarima = SARIMAX(data['Close'], order=(1,1,1), seasonal_order=(1,1,0,12))
result_sarima = model_sarima.fit()
forecast_sarima = result_sarima.forecast(30)


# In[20]:


from prophet import Prophet

df = data.reset_index()[['Date', 'Close']]
df.columns = ['ds', 'y']

model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=30)
forecast_ph= model.predict(future)
# Calculate RMSE for Prophet
rmse_prophet = np.sqrt(mean_squared_error(data['Close'][-30:], forecast_ph['yhat'][-30:]))  # Calculate RMSE for the last 30 data points
import matplotlib.pyplot as plt

model.plot(forecast_ph)


# In[21]:


from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# Create sequences
def create_dataset(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled)
X = X.reshape(X.shape[0], X.shape[1], 1)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)


# Split into train/test
train_size = int(len(scaled) * 0.8)
train, test = scaled[:train_size], scaled[train_size:]

# Create sequences
def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X_train, y_train = create_dataset(train, time_step)
X_test, y_test = create_dataset(test, time_step)

# Reshape for LSTM input (samples, time_steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1], 1)


# Train
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Predict
train_predict = model.predict(X_train)
test_predict  = model.predict(X_test)

# Inverse Scaling
train_predict = scaler.inverse_transform(train_predict)
test_predict  = scaler.inverse_transform(test_predict)

# Plot
plt.plot(data.index, data['Close'], label='Actual') # Changed data.values to data['Close']
plt.plot(data.index[time_step:len(train_predict)+time_step], train_predict, label='Train Predict')
plt.plot(data.index[len(train_predict)+(time_step*2)+1:len(data)-1], test_predict, label='Test Predict')
plt.title('LSTM Forecast')
plt.legend()
plt.show()


# In[ ]:


# 📏 LSTM Forecasting
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data[['Close']])

X, y = [], []
for i in range(60, len(data_scaled)):
    X.append(data_scaled[i-60:i, 0])
    y.append(data_scaled[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dense(1))

model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.fit(X, y, epochs=10, batch_size=32)

# Make predictions for LSTM
# Define scaled_data here, before it's used
scaled_data = scaler.fit_transform(data[['Close']].values)  # Reshape if necessary
train_size = int(len(scaled_data) * 0.8)
train, test = scaled_data[:train_size], scaled_data[train_size:]


def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
_, X_test = create_dataset(test, time_step) # Recalculate X_test
# The original line causing the error:
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Reshape X_test to 3D, but with only one feature
X_test = X_test.reshape(X_test.shape[0], 1, 1)


inputs = data_scaled[len(data_scaled) - len(X_test) - 60:]
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
X_test_lstm = []
for i in range(60, inputs.shape[0]):
    X_test_lstm.append(inputs[i - 60:i, 0])
X_test_lstm = np.array(X_test_lstm)
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

lstm_predictions = model_lstm.predict(X_test_lstm)
lstm_predictions = scaler.inverse_transform(lstm_predictions)


# Calculate RMSE for LSTM
rmse_lstm = np.sqrt(mean_squared_error(data['Close'][-30:], lstm_predictions[-30:]))  # Calculate RMSE for the last 30 data points


# In[ ]:


import streamlit  as st
# 📊 Streamlit Dashboard
st.title("Stock Market Forecast Dashboard")
st.line_chart(data['Close'])
st.subheader("ARIMA Forecast")
st.line_chart(forecast)
st.subheader("SARIMA Forecast")
st.line_chart(forecast_sarima)
st.subheader("Prophet Forecast")
st.line_chart(forecast_ph[['ds', 'yhat']].set_index('ds').tail(30))

st.subheader("Model Accuracy (RMSE)")
st.write(f'ARIMA RMSE: {rmse_arima:.2f}')
st.write(f'SARIMA RMSE: {rmse_sarima:.2f}')
st.write(f'Prophet RMSE: {rmse_prophet:.2f}')
st.write(f'LSTM RMSE: {rmse_lstm:.2f}')

st.success("All forecasts generated successfully!")


# In[ ]:





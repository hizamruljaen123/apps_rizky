# Cryptocurrency Price Prediction: DMA vs GRU

## Overview
This project predicts cryptocurrency prices using two methods:
1. **DMA (Double Moving Average)** – a statistical method based on averaging past prices.
2. **GRU (Gated Recurrent Unit)** – a deep learning model suitable for time series prediction.

The project compares the performance of both methods in terms of prediction accuracy.

## Features
- Input: Historical crypto prices (e.g., Bitcoin, Ethereum)  
- Output: Predicted prices  
- Evaluation: MSE, RMSE, MAE for model comparison

## Steps
1. **Data Collection**: Gather historical price data from APIs (e.g., CoinGecko, Binance)  
2. **Preprocessing**: Fill missing values, normalize prices  
3. **DMA Prediction**: Use simple and double moving averages  
4. **GRU Prediction**: Train GRU model with sequences of past prices  
5. **Comparison**: Evaluate both models and visualize results

## Python Example

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import GRU, Dense
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('crypto_prices.csv')  # columns: date, close
prices = data['close'].values

# --- DMA Prediction ---
def dma_predict(prices, short_window=5, long_window=20):
    short_ma = pd.Series(prices).rolling(window=short_window).mean()
    long_ma = pd.Series(prices).rolling(window=long_window).mean()
    dma = (short_ma + long_ma) / 2
    return dma.fillna(method='bfill').values

dma_pred = dma_predict(prices)

# --- GRU Prediction ---
# Scale data
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices.reshape(-1,1))

# Create sequences
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_sequences(prices_scaled, seq_length)
train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# GRU Model
model = Sequential()
model.add(GRU(50, input_shape=(seq_length,1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# Predict
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# Evaluate
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
print("GRU MSE:", mse)

# Plot comparison
plt.figure(figsize=(12,6))
plt.plot(data['close'], label='Actual Price')
plt.plot(range(len(dma_pred)), dma_pred, label='DMA Prediction')
plt.plot(range(train_size+seq_length, len(prices)), y_pred_rescaled, label='GRU Prediction')
plt.legend()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import datetime
import tensorflow as tf

def predict_usdt_price(config):
    # Set GPU usage based on config
    if config['use_gpu']:
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        tf.config.set_visible_devices([], 'GPU')

    # Load and preprocess data
    file_path = config['file_path']
    data = pd.read_excel(file_path, engine='openpyxl')
    data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d/%m/%Y')
    for column in ['Terakhir_IDR', 'Pembukaan_IDR', 'Tertinggi_IDR', 'Terendah_IDR']:
        data[column] = data[column].astype(float)
    data.set_index('Tanggal', inplace=True)
    data = data.sort_index()

    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Terakhir_IDR']])

    # Create dataset
    def create_dataset(dataset, look_back=config['look_back']):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    X, y = create_dataset(scaled_data, config['look_back'])
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split data
    train_size = int(len(X) * config['train_split'])
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build GRU model
    model = Sequential()
    model.add(GRU(config['gru_units'], return_sequences=True, input_shape=(config['look_back'], 1)))
    model.add(Dropout(0.2))  # Adding dropout to prevent overfitting
    model.add(GRU(config['gru_units']))
    model.add(Dropout(0.2))  # Adding dropout to prevent overfitting
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss='mean_squared_error')

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=config['patience'], restore_best_weights=True)
    try:
        history = model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'],
                            validation_split=config['validation_split'], callbacks=[early_stopping], verbose=1)
    except tf.errors.InvalidArgumentError as e:
        print("Error during model training. Falling back to CPU.")
        tf.config.set_visible_devices([], 'GPU')
        model = Sequential()
        model.add(GRU(config['gru_units'], return_sequences=True, input_shape=(config['look_back'], 1)))
        model.add(Dropout(0.2))
        model.add(GRU(config['gru_units']))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss='mean_squared_error')
        history = model.fit(X_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'],
                            validation_split=config['validation_split'], callbacks=[early_stopping], verbose=1)

    # Predict future days
    def predict_future_days(model, last_data, future_days, scaler, look_back):
        last_sequence = last_data[-look_back:].reshape((1, look_back, 1))
        future_predictions = []
        
        for _ in range(future_days):
            next_pred = model.predict(last_sequence)
            future_predictions.append(next_pred[0, 0])
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_pred[0, 0]
        
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        return scaler.inverse_transform(future_predictions)

    future_predictions_gru = predict_future_days(model, scaled_data, config['future_days'], scaler, config['look_back'])

    # Update data with GRU predictions
    data = data.copy()  # Avoid SettingWithCopyWarning
    future_dates = [data.index[-1] + datetime.timedelta(days=i+1) for i in range(config['future_days'])]
    future_df = pd.DataFrame({
        'GRU_Prediction': future_predictions_gru.flatten()
    }, index=future_dates)

    # Combine historical data with predictions
    historical_data = data[['Terakhir_IDR']].copy()
    historical_data = historical_data.rename(columns={'Terakhir_IDR': 'Actual_Price'})
    combined_data = pd.concat([historical_data, future_df])

    # DMA Prediction
    def predict_dma(data, future_days, short_window, long_window):
        data['GRU_Prediction'] = np.nan
        pred_index = data.index[-config['look_back']:].union(future_df.index)
        pred_values = np.concatenate((scaled_data[-config['look_back']:].flatten(), future_predictions_gru.flatten()))
        data = data.reindex(pred_index)
        data.loc[pred_index, 'GRU_Prediction'] = pred_values

        data['SMA'] = data['GRU_Prediction'].rolling(window=short_window, min_periods=1).mean()
        data['LMA'] = data['GRU_Prediction'].rolling(window=long_window, min_periods=1).mean()
        data['DMA_Prediction'] = (data['SMA'] - data['LMA']) + data['GRU_Prediction']
        
        future_dma_predictions = []
        last_data = data.copy()
        
        for _ in range(future_days):
            new_date = last_data.index[-1] + datetime.timedelta(days=1)
            last_sma = last_data['SMA'].iloc[-1]
            last_lma = last_data['LMA'].iloc[-1]
            last_gru_prediction = last_data['GRU_Prediction'].iloc[-1]
            new_dma = (last_sma - last_lma) + last_gru_prediction
            future_dma_predictions.append(new_dma)
            new_data = pd.DataFrame({'GRU_Prediction': [new_dma]}, index=[new_date])
            last_data = pd.concat([last_data, new_data])
            last_data['SMA'] = last_data['GRU_Prediction'].rolling(window=short_window, min_periods=1).mean()
            last_data['LMA'] = last_data['GRU_Prediction'].rolling(window=long_window, min_periods=1).mean()
        
        return np.array(future_dma_predictions)

    future_predictions_dma = predict_dma(data, config['future_days'], config['short_window'], config['long_window'])

    # Combine predictions
    future_df['DMA_Prediction'] = future_predictions_dma
    future_df['Combined_Prediction'] = (future_df['GRU_Prediction'] + future_df['DMA_Prediction']) / 2

    # Plot results
    plt.figure(figsize=(15, 7))
    plt.plot(combined_data.index, combined_data['Actual_Price'], label='Harga Aktual', color='blue')
    plt.plot(future_df.index, future_df['GRU_Prediction'], label='Prediksi GRU', color='red', linestyle='--')
    plt.plot(future_df.index, future_df['DMA_Prediction'], label='Prediksi DMA', color='green', linestyle='--')
    plt.plot(future_df.index, future_df['Combined_Prediction'], label='Prediksi Kombinasi', color='purple', linestyle='-.')
    plt.axvline(x=data.index[-1], color='black', linestyle='--', label='Batas Prediksi')
    plt.title('Prediksi Harga USDT')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga (IDR)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save results
    combined_data.to_excel(config['output_file'])

    return combined_data

# Contoh konfigurasi
config = {
    'file_path': 'main_data.xlsx',
    'output_file': 'USDT_Predictions_Configurable.xlsx',
    'look_back': 30,
    'future_days': 30,
    'train_split': 0.8,
    'gru_units': 50,
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.2,
    'patience': 10,
    'short_window': 5,
    'long_window': 20,
    'use_gpu': False  # Set this to False to force CPU usage
}

# Jalankan prediksi
results = predict_usdt_price(config)
print(results.tail())

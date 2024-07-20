import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

# Load the provided Excel file
file_path = 'main_data.xlsx'
data = pd.read_excel(file_path)

# Convert the 'Tanggal' column to datetime format
data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d/%m/%Y')

# Sort data by date
data = data.sort_values('Tanggal')

# Prepare data for candlestick chart
ohlc = data[['Tanggal', 'Pembukaan_IDR', 'Tertinggi_IDR', 'Terendah_IDR', 'Terakhir_IDR']].copy()
ohlc = ohlc.rename(columns={
    'Tanggal': 'Date',
    'Pembukaan_IDR': 'Open',
    'Tertinggi_IDR': 'High',
    'Terendah_IDR': 'Low',
    'Terakhir_IDR': 'Close'
})
ohlc.set_index('Date', inplace=True)

# Plot candlestick chart
mpf.plot(ohlc, type='candle', style='charles', title='Candlestick Chart of USDT Tether Coin',
         ylabel='Harga (IDR)', volume=False)

plt.show()

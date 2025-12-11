import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model('./models/Stock Predictions Model.keras')

st.header('Stockify - Stock Market Price Predictor')

stock = st.text_input('Enter Stock Symbol', '')
start = '2012-01-01'
end = '2022-12-31'

# Check if stock symbol is empty or just whitespace
if not stock or stock.strip() == '':
    st.warning('⚠️ Please enter a valid stock symbol to get predictions.')
    st.info('💡 **US Markets:** GOOG (Google), AAPL (Apple), MSFT (Microsoft), TSLA (Tesla)')
    st.info('💡 **Indian NSE:** INFY.NS (Infosys), TCS.NS (TCS), RELIANCE.NS (Reliance)')
    st.info('💡 **Indian BSE:** 500209.BO (Infosys), 532540.BO (TCS), 500325.BO (Reliance)')
    st.markdown('---')
    st.markdown('''
    **📊 Supported Markets:**
    - **US Stocks**: Use ticker symbol directly (e.g., AAPL, GOOGL)
    - **Indian NSE (National Stock Exchange)**: Add `.NS` suffix (e.g., INFY.NS)
    - **Indian BSE (Bombay Stock Exchange)**: Add `.BO` suffix (e.g., 500209.BO)
    ''')
    st.stop()

# Download stock data
try:
    data = yf.download(stock.strip().upper(), start, end)
    
    # Check if data was successfully downloaded
    if data.empty:
        st.error(f'❌ No data found for stock symbol: {stock.strip().upper()}')
        st.info('💡 Please check if the stock symbol is correct and try again.')
        st.stop()
        
except Exception as e:
    st.error(f'❌ Error downloading data for {stock.strip().upper()}: {str(e)}')
    st.info('💡 Please check your internet connection and stock symbol.')
    st.stop()

st.subheader('Stock Data')
st.write(data)

# Check if there's enough data for analysis
if len(data) < 200:
    st.error(f'❌ Insufficient data for analysis. Only {len(data)} days of data available.')
    st.info('💡 The model requires at least 200 days of data for accurate predictions. Please try a different stock symbol or date range.')
    st.stop()

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)
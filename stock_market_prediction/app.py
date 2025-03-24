import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st

model = load_model('/Users/khoinguyen/Documents/GitHub/ML-with-Deploy/stock_market_prediction/Stock_prediction_model.keras')

st.header('Stock Price Predictor')

stock = st.text_input('Enter a stock ticker symbol', 'GOOG')
start = '2014-01-01'
end = '2024-12-31'

data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[:int(len(data)* 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)* 0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_train], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
    
x, y = np.array(x), np.array(y)
from pytrends.request import TrendReq
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from gensim.models import KeyedVectors
import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import time
import json

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# pd.set_option('future.no_silent_downcasting', True)

with st.sidebar:
    st.title("Settings")
    with open(r'C:\Users\nancy\Downloads\archive\company_tickers_exchange.json', 'r') as f:
      data = json.load(f)
      company_names = [entry[2] for entry in data['data']]
      print(company_names)
    stock = st.text_input("Stock Ticker", value="NVDA")
    n = st.number_input("Number of trend indicators", min_value=5, max_value=20, value=8)
    start = st.button("Start Training")

# find keywords
# Make sure you download the model from: https://code.google.com/archive/p/word2vec/
model = KeyedVectors.load_word2vec_format(r"C:\Users\nancy\Documents\GoogleNews-vectors-negative300.bin", binary=True)

# Find similar terms to 'Tesla'
similar_words = model.most_similar("NVIDIA", topn=10)
keywords = [word for word, _ in similar_words]

ticker = yf.Ticker("NVDA")
df_stock = ticker.history(period="1y", interval="1d")
df_stock = df_stock['Close'].to_frame()
print(df_stock.head())
close_values = df_stock['Close'].values
# close_values = close_values[3:]
print(close_values[:5])
# date = df_stock['date'].values[0]

pytrends = TrendReq(hl='en-US', tz=360,
                    timeout=(10,25),
                    # proxies=['https://47.236.224.32:8080',],
                    # retries=2, backoff_factor=0.1, requests_args={'verify':False}
                    )

# keywords = [["Tesla"], ["TSLA"], ["Model Y"], ["electric cars"]]

X_values = []
i=0
while i<len(keywords):
    kw = keywords[i]
    pytrends.build_payload([kw], timeframe='today 12-m')
    try:
        df = pytrends.interest_over_time()
        stock_values = df[kw[0]].values
        X_values.append(stock_values)
        i+=1
    except Exception as e:
        if('429' in str(e)):
            print(f"Rate limit exceeded for {kw}. Retrying after 10 seconds...")
            time.sleep(20)
            continue
    print(kw, stock_values[:5])
    time.sleep(15)
print(df.head())

X_train = np.column_stack(X_values)
# remove first row
X_train = X_train[3:]

# format y values (yfinance data)
Y_values = []
weekly_sum = 0.0
for i in range(len(close_values)):
    weekly_sum += close_values[i]
    if i % 5 == 4:
        Y_values.append(weekly_sum / 7)   # or just weekly_sum
        weekly_sum = 0.0

print(Y_values)

Y_train = np.array(Y_values)
print(X_train.shape, Y_train.shape)
print(X_train[:5], Y_train[:5])

# train
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)   # predicting a single continuous return
])

model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])

history = model.fit(
    X_train, Y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=8,
    # callbacks=[…]       # e.g. EarlyStopping(monitor='val_loss', patience=10)
)

X_next = np.array([[44, 50, 90, 20]])  # Example input for prediction
# Forecast:
y_pred = model.predict(X_next)
print(y_pred)
import streamlit as st
import json
from utils import TrendAnalyzer

with st.sidebar:
    st.title("Settings")
    with open(r'C:\Users\nancy\Downloads\archive\company_tickers_exchange.json', 'r') as f:
      data = json.load(f)
      company_names = [entry[2] for entry in data['data']]
      print(company_names)
    stock = st.text_input("Stock Ticker", value="NVDA")
    n = st.number_input("Number of trend indicators", min_value=5, max_value=20, value=8)
    start = st.button("Start Training")
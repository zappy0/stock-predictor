import streamlit as st

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import json
import matplotlib.dates as mdates
import altair as alt
import base64
import itertools
import datetime
from datetime import date
import plotly.graph_objects as go

from tensorflow import keras


st.set_page_config(page_title="Stock Predictor",
                   initial_sidebar_state="collapsed")

tabs = ["Application"]
page = st.sidebar.radio("Tabs", tabs)

@st.cache(persist=False, suppress_st_warning=True, show_spinner=True, allow_output_mutation=True)
def load_csv(input_metric):
    df_input = None
    df_input = pd.DataFrame()
    df_input = pd.read_csv(input_metric, sep=',', engine='python', encoding='utf-8',
                           parse_dates=True)
    return df_input.copy()


# def prep_data(df):

    # # df_input = df.rename({date_col: "ds", metric_col: "y"},errors='raise', axis=1)
    # # st.markdown(
    # #     "The selected date column is now labeled as **ds** and the values columns as **y**")
    # # df_input = df_input[['ds', 'y']]
    # df_input = df.sort_values(by='ds', ascending=True)
    # df_input['ds'] = pd.to_datetime(df_input['ds'])
    # df_input['y'] = df_input['y'].astype(float)
    # return df_input.copy()


if page == "Application":

    st.title('Compare Stock Forecasts')

    st.write(
        'Select and compare stock forecasts')
    
    # st.markdown(
    #     """The forecasting library used is **[Prophet](https://facebook.github.io/prophet/)**.""")
    # caching.clear_cache()
    
    # df = pd.DataFrame()

    st.subheader('Select Stocks')
    dicttickers = {"Tesla":"TSLA", "Amazon":"AMZN", "Moderna Inc":"MRNA", "Apple":"AAPL", "Microsoft":"MSFT"}
    
    # tickers = st.multiselect('', ['Tesla', 'Amazon', 'Moderna Inc', 'Apple', 'Microsoft','SBI'],['Tesla', 'Amazon'])
    
    tickers = st.multiselect('', dicttickers.keys(),['Tesla', 'Amazon'])    
    
    # st.write('You selected:', tickers)
    
    dataframes=[]
    
    for ticker in tickers:
         
        brk = yf.Ticker(dicttickers.get(ticker))

        hist = brk.history(period="max", auto_adjust=True)

        df = pd.DataFrame()
        
        df=hist.copy()
        dataframes.append((df,ticker))
        
    
    # df = prep_data(df)

    if st.checkbox('Chart Current data', key='show'):
        with st.spinner('Plotting data..'):
            
            fig = go.Figure(layout_xaxis_range=['2020-01-01',date.today()])
            
            for df in dataframes:
                fig = fig.add_trace(go.Scatter(x=df[0].index,y = df[0]["Close"], name = df[1]))
            
            # fig.show()
            st.plotly_chart(fig)
    
            
            
            
            
            
            
            

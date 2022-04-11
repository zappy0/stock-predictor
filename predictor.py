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
from datetime import date,timedelta
import plotly.graph_objects as go
import sklearn
from sklearn.preprocessing import MinMaxScaler
scalerl=[]
import keras


st.set_page_config(page_title="Stock Predictor",
                   initial_sidebar_state="collapsed")

tabs = ["Application"]
page = st.sidebar.radio("Tabs", tabs)

@st.cache(persist=False, suppress_st_warning=True, show_spinner=True, allow_output_mutation=True)

def hehe():
    print("Started")

if page == "Application":

    st.title('Compare Stock Forecasts')

    st.write(
        'Select and compare stock forecasts')
    
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
        dataframes.append([df,ticker])
    
    for i in range(len(tickers)):
        scalerl.append(MinMaxScaler(feature_range=(0,1)))
        
    # st.write(len(scalerl)) 
    
    # df = prep_data(df)

    # if st.checkbox('Chart Current data', key='show'):
    #     with st.spinner('Plotting data..'):
            
    fig = go.Figure(layout_xaxis_range=['2020-01-01',date.today()])
    
    for df in dataframes:
        fig = fig.add_trace(go.Scatter(x=df[0].index,y = df[0]["Close"], name = df[1]))
    
    # fig.show()
    st.plotly_chart(fig)
    
    if len(tickers)!=0:
        sdfs=[]
        i=0
        for df in dataframes:
            dft=df[0].reset_index()['Close']
            dft=scalerl[i].fit_transform(np.array(dft).reshape(-1,1))
            dft=dft[-100:,:1].reshape(1,-1)
            x_input=dft
            dft=list(x_input)
            dft=dft[0].tolist()
            sdfs.append([dft,df[1]])
            # print(df[1])
            # print(dft)
            i+=1
        
        predictions=[]
        n_steps=100
        
        # print(dicttickers.get(sdfs[0][1]))
        for xin in sdfs:
            path="static/"+dicttickers.get(xin[1])+"_model.h5"
            # print("Ticker:::::")
            # print(dicttickers.get(xin[1]))
            model = keras.models.load_model(path)
            i=0
            out=[]
            while(i<30):
            
                if(len(xin[0])>100):
                    #print(xin[0])
                    x_input=np.array(xin[0][1:])
                    # print("{} day input {}".format(i,x_input))
                    x_input=x_input.reshape(1,-1)
                    x_input = x_input.reshape((1, n_steps, 1))
                    #print(x_input)
                    yhat = model.predict(x_input, verbose=0)
                    # print("{} day output {}".format(i,yhat))
                    # print(xin[0])
                    # type(xin[0])
                    xin[0].extend(yhat[0].tolist())
                    xin[0]=xin[0][1:]
                    #print(xin[0])
                    out.extend(yhat.tolist())
                    i=i+1
                else:
                    x_input = x_input.reshape((1, n_steps,1))
                    yhat = model.predict(x_input, verbose=0)
                    # print(yhat[0])
                    xin[0].extend(yhat[0].tolist())
                    # print(len(xin[0]))
                    out.extend(yhat.tolist())
                    i=i+1
        
            # print(out)
            predictions.append([out,xin[1]])
        
        # day_new=np.arange(1,101)
        # day_pred=np.arange(101,131)
        # print(predictions[0][0])
        # print(dataframes[0].index) 
        preddf=[]
        i=0
        dataframes[0][0].reset_index()
        start=pd.to_datetime(dataframes[0][0].index[dataframes[0][0].shape[0]-1],format='%Y-%m-%d') 
        
        finaldfs=[]
        # print(predictions)              #correct
        j=0
        for prediction in predictions:
            # df[0].reset_index()
            v2_data=pd.DataFrame(columns=['Date','Prediction'])
            # v2_data.loc[0]=pd.to_datetime(df[0].index[df[0].shape[0]-1],format='%Y-%m-%d')   
            v2_data.loc[0]=start 
            # print(v2_data)
            
            date=v2_data.iloc[0,0]
            date += timedelta(days=1)
            for i in range(30):
                v2_data.loc[i] = [date,0]
                date += timedelta(days=1)
                
            v2_data.index=v2_data['Date']
            v2_data.drop(['Date'],axis=1,inplace=True)    

            # print("Predict::::::::::::::")
            # print(v2_data,prediction[1])
            # print("TICCCCK",prediction[1])
            # print(prediction[0])
            # st.write(j)
            v2_data['Prediction']=scalerl[j].inverse_transform(prediction[0])
            
            # print("Predict::::::::::::::")
            # print(prediction[1],v2_data)
            
            finaldfs.append([v2_data,prediction[1]])
            j+=1
        
        ############
        # print(finaldfs[0][1],finaldfs[0][0])
    else:
        st.write("Please Select a stock to predict")
    
    st.subheader('Predictions')
    fig2 = go.Figure(layout_xaxis_range=[date.today(),date.today()+timedelta(days=40)])
    
    if len(tickers)>0:
        for finaldf in finaldfs:
                
            fig2 = fig2.add_trace(go.Scatter(x=finaldf[0].index,y = finaldf[0]["Prediction"], name = finaldf[1]))
        
        st.plotly_chart(fig2)
    
    for finaldf in finaldfs:
        st.write(finaldf[1])
        st.dataframe(finaldf[0][1:8]) 
    

#Inspiration
#https://stackoverflow.com/questions/68601505/how-to-feed-an-lstm-gru-model-multiple-independent-time-series

#General
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from datetime import datetime
import tqdm
import os
from dateutil.relativedelta import relativedelta

#Scrape
from pandas_datareader import data as wb
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

#ML
#importing the packages 
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

SEQ_LEN = 60

TRAIN_TICKERS = ['MRVL'] #['SPY', 'GOOG', 'SCHG', 'VNQ', 'XRT', 'IBUY']

def create_model(SEQ_LEN = 60):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=128, return_sequences=True, input_shape=(SEQ_LEN,1)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(units=512,return_sequences=True))
    lstm_model.add(Dropout(0.2))
    """
    lstm_model.add(LSTM(units=256,return_sequences=True))
    lstm_model.add(Dropout(0.2))
    """
    lstm_model.add(LSTM(units=128))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1))
    return lstm_model

def train(model):
    #grab ticker data
    start = (datetime.today() - relativedelta(years=1)).strftime('%Y-%m-%d')
    today = datetime.today().strftime('%Y-%m-%d')
    price_data = []
    for t in TRAIN_TICKERS:
        prices = yf.download(t, start=start, end=today)[['Open','Adj Close']]
        price_data.append(prices.assign(ticker=t)[['ticker', 'Open', 'Adj Close']])
    df = pd.concat(price_data)
    df.reset_index(inplace=True)
    #convert to df
    #performs an 80/20 train test split on a dataframe
    ntrain = 80
    df_train = df.head(int(len(df)))
    ntest = -80
    df_test = df.tail(int(len(df)*(ntest/100)))

    #dataframe creation
    seriesdata = df.sort_index(ascending=True, axis=0)
    new_seriesdata = pd.DataFrame(index=range(0,len(df)),columns=['Date','Adj Close'])
    length_of_data=len(seriesdata)
    for i in range(0,length_of_data):
        new_seriesdata['Date'][i] = seriesdata['Date'][i]
        new_seriesdata['Adj Close'][i] = seriesdata['Adj Close'][i]

    #setting the index again
    new_seriesdata.index = new_seriesdata.Date
    new_seriesdata.drop('Date', axis=1, inplace=True)

    #creating train and test sets this comprises the entire data’s present in the dataset
    myseriesdataset = new_seriesdata.values
    totrain = myseriesdataset #myseriesdataset[0:(len(myseriesdataset)+1),:]
    tovalid = myseriesdataset[(len(myseriesdataset)+1):,:]

    #converting dataset into x_train and y_train
    scalerdata = MinMaxScaler(feature_range=(0, 1))
    scale_data = scalerdata.fit_transform(myseriesdataset)
    x_totrain, y_totrain = [], []
    length_of_totrain=len(totrain)
    #x_totrain becomes list of SEQ_LEN days up til day
    for i in range(SEQ_LEN,length_of_totrain):
        x_totrain.append(scale_data[i-SEQ_LEN:i,0])
        y_totrain.append(scale_data[i,0])
    x_totrain, y_totrain = np.array(x_totrain), np.array(y_totrain)
    x_totrain = np.reshape(x_totrain, (x_totrain.shape[0],x_totrain.shape[1],1))

    earlystop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=5, verbose=0)
    model.compile(loss='mean_squared_error', optimizer='adam')
    #model.fit(x_totrain, y_totrain, epochs=100, batch_size=64, verbose=1)

    return scalerdata

def predict_close(model, tickers, scalerdata):
    pred = {}
    tickers = tickers.split(" ")
    for t in tickers:
        #grab ticker data
        start = (datetime.today() - relativedelta(years=1)).strftime('%Y-%m-%d')
        today = datetime.today().strftime('%Y-%m-%d')
        price_data = []
        prices = yf.download(t, start=start, end=today)[['Open','Adj Close']]
        price_data.append(prices.assign(ticker=t)[['ticker', 'Open', 'Adj Close']])
        df = pd.concat(price_data)
        df.reset_index(inplace=True)
        #convert to df
        #performs an 80/20 train test split on a dataframe
        ntrain = 80
        df_train = df.head(int(len(df)*(ntrain/100)))
        ntest = -80
        df_test = df.tail(int(len(df)*(ntest/100)))

        #dataframe creation
        seriesdata = df.sort_index(ascending=True, axis=0)
        new_seriesdata = pd.DataFrame(index=range(0,len(df)),columns=['Date','Adj Close'])
        length_of_data=len(seriesdata)
        for i in range(0,length_of_data):
            new_seriesdata['Date'][i] = seriesdata['Date'][i]
            new_seriesdata['Adj Close'][i] = seriesdata['Adj Close'][i]

        #setting the index again
        new_seriesdata.index = new_seriesdata.Date
        new_seriesdata.drop('Date', axis=1, inplace=True)

        seq_length = 1
        price_pred = []
        tostore_test_result = []

        myinputs = new_seriesdata[len(new_seriesdata) - 1 - SEQ_LEN:].values
        myinputs = myinputs.reshape(-1,1)
        myinputs  = scalerdata.transform(myinputs)

        for i in range(SEQ_LEN+1,myinputs.shape[0]+1):
            tostore_test_result.append(myinputs[i-SEQ_LEN:i,0])
        tostore_test_result = np.array(tostore_test_result)
        tostore_test_result = np.reshape(tostore_test_result,(tostore_test_result.shape[0],tostore_test_result.shape[1],1))
        rescale_dims = (tostore_test_result.shape[0],tostore_test_result.shape[1],1)

        for s in tqdm.tqdm(range(seq_length)):
            #make a prediction for the next day
            myclosing_priceresult = model.predict(tostore_test_result, verbose=0)
            price = scalerdata.inverse_transform(myclosing_priceresult)
            price_pred.append(price[0][0])
            
            #drop oldest value in sequence data, add predicted one in, and reshape to LSTM input dimensions
            tostore_test_result = np.delete(tostore_test_result, 0, axis=1)
            tostore_test_result = np.append(tostore_test_result, myclosing_priceresult)
            tostore_test_result = np.reshape(tostore_test_result,rescale_dims)

        pred[t] = str(round(price_pred[0], 2))

    return pred
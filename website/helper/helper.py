#General
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from datetime import datetime
import tqdm
import os
import json
import flask
from flask import jsonify, make_response
from dateutil.relativedelta import relativedelta
import time

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

SLEEP_DELAY = .05 #For getting rate limited by yfinance
TRIES = 3

def get_data_recent(ticker, years_ago):
    start = (datetime.today() - relativedelta(years=years_ago)).strftime('%Y-%m-%d')
    today = datetime.today().strftime('%Y-%m-%d')

    repeats = 0
    while(repeats < TRIES):
        try:
            prices = yf.download(ticker, start=start, end=today)[['Open','Adj Close']]
            break
        except:
            repeats += 1
            time.sleep(SLEEP_DELAY)
        if repeats == TRIES:
            return {'name' : ticker + " - NO DATA", 'dates':[], 'adj_close':[], 'open':[]}
            

    tD = {
        'name' : ticker,
        'dates' : prices.index.astype(str).tolist(),
        'adj_close' : prices['Adj Close'].tolist(),
        'open' : prices['Open'].tolist()
    }
    
    return tD
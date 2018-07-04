#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 01:02:07 2018

@author: aditi
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_data(filename):
    
    '''
    This functions reads file in csv format and index is 
    set according to dates.
    '''
    
    df = pd.read_csv(filename,parse_dates=['Date'])
    df = df.dropna()
    df.set_index('Date',inplace=True)
    return df



def data_set(dataframe):
    ''' 
    splitting into test ,training and cross validation datasets
    in ratio of 8:1:1
    '''
    length = len(dataframe.iloc[:,0])
    train_end = round(length * 0.80)
    test_end = round(length * 0.90)
    return [
            dataframe.iloc[: train_end, : ],
            dataframe.iloc[train_end:test_end, :],
            dataframe.iloc[test_end: ,:]
           ]


def feature_scaling(df):
    '''
    Feature scaling.It is important 
    to note that scaling should be done according to training set
    '''
    [train ,test, cross] = data_set(df)
    
    scaler = MinMaxScaler()
    scaler.fit(train.values) 
    train.loc[:,:] = scaler.transform(train.values)
    test.loc[:,:] = scaler.transform(test.values)
    cross.loc[:,:] = scaler.transform(cross.values)
    
    return [train, test, cross] # This function returns dataframes



def features(df):
    
    '''
    This function generates a set of features from given data 
    and returns features(X) as a dataframe .    
    We aim to predict the stock price for next day 
    ''' 
    df['High-Low'] = df['High']-df['Low']
    df['PCT_change'] = (df['Close'] - df['Open'])/df['Open'] * 100
    df['WILR'] = (df['High']- df['Close'])/(df['High']- df['Low'])*100
     
    df['MAV3'] = (df.loc[:,'Close']).rolling(window =3).mean() # moving day average for 3-day periods
    df['MAV5'] = (df.loc[:,'Close']).rolling(window =5).mean() # moving day average for 5-day periods

    df['1DayW'] = df['Adj Close'].shift(-1) 
    ''' creates column for next day prices. We aim to predict 
        this for data available till one day before '''
    
    
    df =df.dropna() 
    
    X = df.loc[:,'Adj Close':'MAV5']
    
    y =  pd.DataFrame(df,columns = ['1DayW'])   
    return [X , y]
    

  


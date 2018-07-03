#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 01:02:07 2018

@author: aditi
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler



# our data is already set in order od dates
def get_data(filename):
# reading data from CSV file and parsing date to corrrect format
    df = pd.read_csv(filename, parse_dates=['Date'])
# to drop out missing information laabelled as NaN and redundant column
    df = df.dropna()
    df = df.drop(columns = ['Adj Close'])
# to ser index according to Date 
    df.set_index('Date',inplace=True)
    return df

# splitting into test ,training and cross validation datasets
def data_set(dataframe):
    #return type is pandacoredataframes
    length = len(dataframe.iloc[:,0])
    train_end = round(length * 0.80)
    test_end = round(length * 0.90)
    return [
            dataframe.iloc[: train_end, : ],
            dataframe.iloc[train_end:test_end, :],
            dataframe.iloc[test_end: ,:]
           ]


def feature_scaling(df):
    [train ,test, cross] = data_set(df)
    
    scaler = MinMaxScaler()
    # scaling only to be done according to the training set .
    scaler.fit(train.values)
    train.loc[:,:] = scaler.transform(train.values)
    test.loc[:,:] = scaler.transform(test.values)
    cross.loc[:,:] = scaler.transform(cross.values)
    
    # we have returned panda dataframes here as well
    return [train, test, cross]




def features(df):
    
    df['High-Low'] = df['High']-df['Low']
    df['PCT_change'] = (df['Close'] - df['Open'])/df['Open'] * 100
    df['WILR'] = (df['High']- df['Close'])/(df['High']- df['Low'])*100
    df['MAV3'] = (df.loc[:,'Close']).rolling(window =3).mean()
    df['MAV5'] = (df.loc[:,'Close']).rolling(window =5).mean()
    df =df.dropna()
    target =  pd.DataFrame(df,columns = ['Close'])
    X = df.loc[:,'Volume':]
    return [X , target] 
    


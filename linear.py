#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 01:09:04 2018

@author: aditi
"""
'''
This program trains linear model and prints
confidence on both test and cross validation set '
The plot_result function prints the actual and 
predicted price for the next day
'''
import data_prep 
from sklearn import linear_model
import pandas as  pd

from matplotlib import pyplot as plt ,style


pd.options.mode.chained_assignment = None  # default='warn'
style.use('ggplot')

def train(filename):
    df = data_prep.get_data(filename)
    [X, y] = data_prep.features(df)
    [X_train, X_test, X_cross] = data_prep.feature_scaling(X)
    [y_train, y_test, y_cross] = data_prep.data_set(y)
    lm = linear_model.LinearRegression()
    
    model = lm.fit(X_train.values,y_train.values) # training model on training set
    
    predictions = model.predict(X_test.values)
    print("confidence on test set is ",lm.score(X_test.values, y_test.values)*100)    
    predictions = model.predict(X_cross.values)
    print("confidence on cross validation set is ",lm.score(X_cross.values ,y_cross.values)*100)
    y_cross['predictions'] = predictions
    
    return y_cross
    

def plot_result(filename):
    
    y_cross = train(filename)
    y_cross['1DayW'].plot()
    plt.legend(loc=4) 
    y_cross['predictions'].plot(color ='b')
    plt.legend(loc=4)  
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
    

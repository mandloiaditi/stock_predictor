#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 01:09:04 2018

@author: aditi
"""

import data_prep 
from sklearn import linear_model
import pandas as  pd
import numpy as np
from matplotlib import pyplot as plt ,style
import sys 

def train():
    df = data_prep.get_data("GOOG.csv")
    [X, y] = data_prep.features(df)
    [X_train, X_test, X_cross] = data_prep.feature_scaling(X)
    [y_train, y_test, y_cross] = data_prep.data_set(y)
    lm = linear_model.LinearRegression()
    model = lm.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print("confidence on test set is ",lm.score(X_test, y_test))    
    predictions = model.predict(X_cross)
    print("confidence on cross_validation set is ",lm.score(X_cross, y_cross))
    y_cross['predictions'] = predictions
    y_cross['Close'].plot()
    y_cross['predictions'].plot()
    plt.legend(loc=4)  
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
    
#def predict(argv[2]):
    
    
    
train()
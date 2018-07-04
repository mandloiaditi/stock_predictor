#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 01:10:41 2018

@author: aditi
"""
'''
This program trains SVR with different kernels to print
confidence on cross validation set and plots the graph 
of svr wth linear kernel
'''

from sklearn.svm import SVR
import data_prep
from matplotlib import pyplot as plt, style
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
style.use('ggplot')

def plot_result(filename) :
    
    df = data_prep.get_data(filename)
    [X, y] = data_prep.features(df)
    
    [X_train, X_test, X_cross] = data_prep.feature_scaling(X)
    [y_train, y_test, y_cross] = data_prep.data_set(y)
    
    
    ''' SVR with different kernels '''
    svr_lin = SVR(kernel= 'linear', C= 1e3) 
    svr_poly = SVR(kernel= 'poly', C= 1e3, degree = 2)
    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.01) 
    
    '''fitting model on training set '''
    svr_rbf.fit(X_train.values, y_train.loc[:,'1DayW'].values) 
    svr_lin.fit(X_train.values, y_train.loc[:,'1DayW'].values)
    svr_poly.fit(X_train.values, y_train.loc[:,'1DayW'].values)
    
    
    lin_score = svr_lin.score(X_cross,y_cross)
    poly_score = svr_poly.score(X_cross,y_cross)
    rbf_score = svr_rbf.score(X_cross,y_cross)
    
    
    print('Confidence score for linear kernel :',lin_score*100)
    print('Confidence score for poly. kernel :',poly_score*100)
    print('Confidence score for rbf kernel :',rbf_score*100)
    
    
    print("\nPlotting graph for rbf kernel:")
    plt.scatter(y_cross.index,y_cross.values ,color = 'black')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.plot(y_cross.index ,svr_rbf.predict(X_cross),color = 'b',label = 'RBF Kernel SVR')
    plt.show()
    
    
    print("\nPlotting graph for linear kernel:")
    plt.scatter(y_cross.index,y_cross.values ,color = 'black')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.plot(y_cross.index ,svr_lin.predict(X_cross),color = 'r',label = 'Linear Kernel SVR')
    plt.show()

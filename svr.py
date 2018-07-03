#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 01:10:41 2018

@author: aditi
"""

from sklearn.svm import SVR
import data_prep
from matplotlib import pyplot as plt

def train() :
    df = data_prep.get_data("GOOG.csv")
    [X, y] = data_prep.features(df)
    [X_train, X_test, X_cross] = data_prep.feature_scaling(X)
    [y_train, y_test, y_cross] = data_prep.data_set(y)
    svr_lin = SVR(kernel= 'linear', C= 1e3) 
    svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) 
    svr_rbf.fit(X_train, y_train.loc[:,'Close']) 
    svr_lin.fit(X_train, y_train.loc[:,'Close'])
    svr_poly.fit(X_train, y_train.loc[:,'Close'])
    
    plt.plot(y_test.index, y_test, color= 'black', label= 'Data') 
    plt.plot(y_test.index, svr_rbf.predict(X_test), color= 'red', label= 'RBF model') 
    plt.plot(y_test.index,svr_lin.predict(X_test), color= 'green', label= 'Linear model') 
    plt.plot(y_test.index,svr_poly.predict(X_test), color= 'blue', label= 'Polynomial model') 
 
    print(svr_lin.score(X_test,y_test))
    print(svr_poly.score(X_test,y_test))
    print(svr_rbf.score(X_test,y_test))
train()
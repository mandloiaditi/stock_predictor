#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 21:39:04 2018

@author: aditi
"""
''' 
This programs compares four predictive models
for stock price predictions : linear, and svr 
using three different kernels and shows the 
result by plotting corresponding graphs.
The prediction is made on the next day closing 
price with data available till the previous day
'''
import os
import sys 
import linear 
import svr



if len(sys.argv) ==1 :
    print('Please enter the csv file path name')
    exit()
elif os.path.exists(sys.argv[1]):
    print('predicting stock prices for',sys.argv[1])
else:
    print("Enter existing filepath")      
    exit()
    
filename =sys.argv[1]

#filename = ./Datasets/GOOG.csv (example)
print("For Linear Regression Model :")

linear.plot_result(filename)

print("\n***********************************\n")

print("For SVR Model :\n")
svr.plot_result(filename)



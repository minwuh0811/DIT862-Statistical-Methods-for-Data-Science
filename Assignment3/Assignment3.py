# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:30:56 2019

@author: bishe
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

N = 15
t = np.reshape(np.array([132.1,133.0,134.0,138.8,145.5,147.0,147.1,149.0,150.0,149.0,152.4,152.6, 153.4, 155.8, 156.0]), (N,1))
X = np.array([[1,26.0],[1,26.5],[1,28.1],[1,28.9],[1,30.1],[1,31.7],[1,32.0],[1,33.1],[1,33.2],[1,33.0],[1,34.1],[1,34.5],[1,35.2],[1,36.0],[1,36.7]], np.float)
fig=plt.figure(figsize=(19,20))
ax1=fig.add_subplot(2,2,1)
ax1.set_xlabel('x', fontsize=18)
ax1.set_ylabel('t', fontsize=18)
ax1.scatter(np.transpose(X)[1],t)
#t_train=t[:8]
#t_test=t[8:]
#X_train=X[:8]
#X_test=X[8:]

def LinearRegression(X,t,max_iter=1000): 
    N=15
    totXt=0
    totX2=0
    Xmean=np.mean(X,axis=0)[1]
    tmean=np.mean(t,axis=0)[0]
    t_pred=[]
    for n in range(N):
        xn=X[n][1]
        tn=t[n][0]
        totX2+=xn*xn
        totXt+=xn*tn
    x2mean=totX2/15    
    Xtmean=totXt/15
    w1=(Xtmean-Xmean*tmean)/(x2mean-Xmean*Xmean)
    w0=tmean-w1*Xmean
    for n in range(N):
        t_pred.append(w0+w1*X[n][1])
    return w0,w1,t_pred

def CoefficientDetermAndMeanSquaredLoss(X, t, w0, w1):
    resSquared=0
    totSquared=0
    tmean=np.mean(t,axis=0)[0]
    for n in range(15):
        resSquared+=(t[n][0]-(w0+w1*X[n][1]))**2
        totSquared+=(t[n][0]-tmean)**2   
    return 1-resSquared/totSquared, resSquared/15   


w0,w1,t_pred=LinearRegression(X,t)
CoeffD, meansq=CoefficientDetermAndMeanSquaredLoss(X,t,w0,w1)
print('Coefficients: w0: %0.2f, w1: %0.2f' %(w0,w1)) 
print('Mean squared error: %.2f' %(meansq))
print('Coefficient of determination: %.2f' %(CoeffD))
ax2=fig.add_subplot(2,2,2)
ax2.scatter(np.transpose(X)[1], t,  color='black')
ax2.plot(np.transpose(X)[1], t_pred, color='blue', linewidth=3)

xnew=[[1],[32.5]]
w=[[w0],[w1]]
tnew=np.dot(np.transpose(xnew),w)
print(tnew)
b=np.linalg.inv(np.dot(np.transpose(X),X))
print(b)
a=meansq*np.dot(np.transpose(xnew),b)
print(a)
c=np.dot(a,xnew)
print(c)

"""
My Assumption 

The relationship is linear, choose a linear model f(x)=w0+w1x

fit the linear model

Evaluate the model by mean squared error and Coefficient of determination

The results are good as I expected the coefficient of determination is 0.97 which means the model can determine 97% of the data

By calculating the mean squared error the variance of the distribution is obtained 1.97 as a parameter for the probabilistic regression model

"""




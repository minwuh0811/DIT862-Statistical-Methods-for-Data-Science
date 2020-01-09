# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:03:00 2019

@author: bishe
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def NumberRand(Number,n,fig):
    randomNumber=np.random.rand(1,Number)
    ax=fig.add_subplot(3,2,n)
    ax.set_xlabel('Random Number', fontsize=18)
    ax.set_ylabel('Count', fontsize=18)
    ax.hist(randomNumber[0],bins=10,edgecolor='w')
    ax.set_title(f'Histogram-RandomNumber {Number}', fontsize=20)
    return
#Generate a set of random numbers
#10000 random numbers
fig=plt.figure(figsize=(19,20))   
NumberRand(10000,1,fig)    
#10000,1000,100,10 random numbers
fig=plt.figure(figsize=(19,20))   
for n,i in zip([10000,1000,100,10],[1,2,3,4]):
        NumberRand(n,i,fig)
        
randomNorm=np.random.normal(0,10,10000)     
ax=fig.add_subplot(3,2,6)
ax.set_xlabel('Random Number', fontsize=18)
ax.set_ylabel('Count', fontsize=18)
ax.hist(randomNorm,bins=10,edgecolor='w')
ax.set_title(f'Histogram-Random Normal distribution 10000', fontsize=20)   
fig.show()



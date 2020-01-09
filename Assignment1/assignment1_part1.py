# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:14:44 2019

@author: bishe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def diffRange(number1,number2,obj):
    diffNumber=[]
    for x in obj:
        if x<=number2 and x>number1:
            diffNumber.append(x)
    return diffNumber

#Load the CSV file into Python.
obj=pd.read_csv('houses.csv',sep=',',header=None)
#print(obj[1])
#Print the properties (mean, median, standard deviation, minimum, and maximum) of the second column  .
print(f"Price\nMax: {np.max(obj[1]):.2f}, Min: {np.min(obj[1]):.2f}, Mean: {np.mean(obj[1]):.2f}, Median: {np.median(obj[1]):.2f}, Standard Deviation: {np.std(obj[1]):.2f}")
#Plot a histogram that shows the distribution of the prices.
fig=plt.figure(figsize=(19,20))
ax1=fig.add_subplot(3,2,1)
ax2=fig.add_subplot(3,2,2)
ax1.set_xlabel('Price', fontsize=18)
ax1.set_ylabel('Price', fontsize=18)
ax1.scatter(obj[1],obj[1])
ax1.ticklabel_format(style='plain')
ax1.set_title('Scatter',fontsize=20)
ax2.hist(obj[1],bins=50)
ax2.set_xlabel('Price', fontsize=18)
ax2.set_ylabel('Price', fontsize=18)
ax2.ticklabel_format(style='plain')
ax2.set_title('Histogram',fontsize=20)       
#Eprices distributions in different ranges
#Range 0-10000000
ax3=fig.add_subplot(3,2,3)
ax3.set_xlabel('Price', fontsize=18)
ax3.set_ylabel('Count', fontsize=18)
ax3.set_title('Range 0-1e7',fontsize=20)
objRange=diffRange(0,1000000,obj[1])
ax3.hist(objRange,bins=50,edgecolor='w')
#Range 10000000-60000000
ax4=fig.add_subplot(3,2,4)
ax4.set_xlabel('Price', fontsize=18)
ax4.set_ylabel('Count', fontsize=18)
ax4.set_title('Range 1e7-6e7',fontsize=20)
objRange=diffRange(1000000,6000000,obj[1])
ax4.hist(objRange,bins=50,edgecolor='w')
#inside and outside London
inLondon=[]
outLondon=[]
for index, row in obj.iterrows():
    if 'LONDON' in row[12]:
        if  row[1]<1000000:
            inLondon.append(row[1])
    else:
        if row[1]<1000000:
            outLondon.append(row[1])
ax5=fig.add_subplot(3,2,5)  
ax5.set_xlabel('Price', fontsize=18)
ax5.set_ylabel('Normalized Count', fontsize=18)
ax5.set_title('Inside and outside London',fontsize=20)
hist,bins=np.histogram(inLondon,bins=50)
widths = np.diff(bins)
ax5.bar(bins[:-1], hist/np.max(hist), widths,label='inLondon',edgecolor='w',alpha=0.7)
hist,bins=np.histogram(outLondon,bins=50)
widths = np.diff(bins)
ax5.bar(bins[:-1], hist/np.max(hist), widths,label='outLondon',edgecolor='w',alpha=0.7)
ax5.legend(loc='upper right',fontsize=18)
#Make a plot that shows the average price per year.
ax6=fig.add_subplot(3,2,6)  
#group=obj[1].groupby(pd.DatetimeIndex(obj[2]).year)
#print(group.mean())
obj[1].groupby(pd.DatetimeIndex(obj[2]).year).mean().plot(ax=ax6)
ax6.set_xlabel('Year', fontsize=18)
ax6.set_ylabel('Price', fontsize=18)
ax6.set_title('Year-Mean Price',fontsize=20)
fig.show()


      

        



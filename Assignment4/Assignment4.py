# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:06:25 2019

@author: bishe
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
ns = 100
s = 10
# generate spherical data 

data1 = np.random.randn(ns, 2)
data2 = np.random.randn(ns, 2) + np.array([s, 0])
data3 = np.random.randn(ns, 2) + np.array([0, s])


# concatenate 
data = np.vstack([data1,data2,data3])
# You may plot the data to see how it looks like 
plt.scatter(data[:, 0], data[:, 1], .8)
plt.show()
K = 5
gmm = mixture.GaussianMixture(n_components=K, n_init=10)
gmm.fit(data)
print(len(data)*gmm.score(data))
y_pred=gmm.predict(data)
colors=[]
for i in y_pred:
    if i==0:
        colors.append('#377eb8')
    elif i==1:
        colors.append('#ff7f00')
    elif i==2:
        colors.append('#4daf4a')
    elif i==3:
        colors.append('#f781bf')
    elif i==4:
        colors.append('#a65628')
        
plt.scatter(data[:, 0], data[:, 1], color=colors)

# calculate the model complexity
def ck(K, D=2):
    return K*D + (K-1) + K*D* (D + 1)/2
print(ck(1))
print(ck(2))
print(ck(3))
print(ck(4))
print(ck(5))
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 00:22:03 2019

@author: bishe
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(19,20)) 
ax1=fig.add_subplot(3,2,1)

"""
Part A
"""
def success(p_success):    
    randomNum = np.random.rand()
    if randomNum<p_success:
        return True
    else:
        return False
        
def exam_score(p_correct, n_instances):
    result=0;
    for n in range(n_instances):
        if success(p_correct):
            result+=1
    return result
    
#this function simulates a random variable with a Bernoulli distribution. 
#print(randomNumList)
result=[]
for n in range(10000):
    result.append(success(0.5))   
obj=pd.DataFrame(result)   
obj[0].value_counts(normalize=True).plot(kind='bar', title='True-False', ax=ax1)

#List
result=[]
#NumPy Array
array = np.arange(100)
#Pandas Series
ser = pd.Series() 
for n in range(100):
    randomP = np.random.rand()
    randomInt=np.random.randint(1,1000)
    #collect the result in list, NumPy array, or Pandas Series.
    result.append(exam_score(randomP,randomInt))


result=[]
# p_correct be 0.8 and n_instances be 20. Run exam_score 10,000 times
for n in range(10000):
    #collect the result in list, NumPy array, or Pandas Series.
    result.append(exam_score(0.8,100))
ax2=fig.add_subplot(3,2,2)
ax2.hist(result, bins=10,edgecolor='w')
ax2.set_xlabel('Number of Question', fontsize=18)
ax2.set_ylabel('Count', fontsize=18)


"""
Part B
"""
#function to calculate the number of attempts the student needed before passing
def number_of_attempts(p_pass):
    number=0
    passExam=False
    while(not passExam):
        if success(p_pass):
            passExam=True
        number+=1     
    return number

# p_pass be 0.4 and run the simulation 10000 times
result=[]
for n in range(10000):
    result.append(number_of_attempts(0.4))
ax3=fig.add_subplot(3,2,3)
ax3.hist(result, bins=10,edgecolor='w')
ax3.set_xlabel('Number of times to Pass', fontsize=18)
ax3.set_ylabel('Count', fontsize=18)


"""
Part C
"""
#return boolean if man return true otherwise return false
def gender(p_man):
    randomNum = np.random.rand()
    if randomNum<p_man:
        return True
    else:
        return False    

# return random number of height and weight according to gender
def randomNumGeneration(man):
    if man:
        height=np.random.normal(140, 15)
        weight=np.random.normal(90, 10)
    else:
        height=np.random.normal(195, 10)
        weight=np.random.normal(60, 5)
    return height,weight

result=[]
for n in range(50):
    result.append(randomNumGeneration(gender(0.4)))
obj=pd.DataFrame(result)
ax4=fig.add_subplot(3,2,4)
ax4.scatter(obj[0],obj[1])
ax4.set_xlabel('Height', fontsize=18)
ax4.set_ylabel('Weight', fontsize=18)





















    
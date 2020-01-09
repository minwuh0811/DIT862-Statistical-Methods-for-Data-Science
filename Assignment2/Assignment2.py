# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:59:50 2019

@author: bishe
"""
from __future__ import division
from codecs import open
import numpy as np
import pandas as pd
from collections import Counter
import scipy.stats
import matplotlib.pyplot as plt

def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    return docs, labels

def train_nb(documents, labels):
    #split_point = int(0.80*len(all_docs))
    #train_docs = all_docs[:split_point]
    #train_labels = all_labels[:split_point]  
    classifyList = pd.DataFrame(
    {'documents': documents,
     'labels': labels,
    })

    return classifyList
#first score is calculated using the number of choices obtained by documents    
def score_doc_label(document, label, classifyList):
    gLabel=classifyList.groupby('labels').get_group(label)['documents']
    labelUnique=len(classifyList['labels'].unique())
    gk=classifyList.groupby('labels').get_group(label).size
    gkTotal=classifyList.size
    #counter the freqencies of each word in a certain class
    freqsgLabel = Counter(w for doc in gLabel for w in doc)
    totalWords=sum(freqsgLabel.values())
    #totalWords=171476
    #m is the number of choices
    m=len(sorted(freqsgLabel))
    #"add-one smoothing"
    n=np.log(float(gk+1)/(gkTotal+labelUnique))
    for word in document:
        n+=np.log(float(freqsgLabel[word]+1)/(totalWords+m))
    return n
#second score is calculated using the number of choices from dictionary 
def score_doc_label_2(document, label, classifyList):
    gLabel=classifyList.groupby('labels').get_group(label)['documents']
    labelUnique=len(classifyList['labels'].unique())
    gk=classifyList.groupby('labels').get_group(label).size
    gkTotal=classifyList.size
    #counter the freqencies of each word in a certain class
    freqsgLabel = Counter(w for doc in gLabel for w in doc)
    #totalWords=sum(freqsgLabel.values())
    totalWords=171476
    #m is the number of choices
    m=len(sorted(freqsgLabel))
    #"add-one smoothing"
    n=np.log(float(gk+1)/(gkTotal+labelUnique))
    for word in document:
        n+=np.log(float(freqsgLabel[word]+1)/(totalWords+m))
    return n
# classify by first smoothing method
def classify_nb(document,classifyList):
    label=classifyList['labels'].unique()
    n=[]
    for classifier in label:
      n.append(score_doc_label(document, classifier, classifyList))
    return label[n.index(max(n))]   
# classify by the second smoothing method
def classify_nb_2(document,classifyList):
    label=classifyList['labels'].unique()
    n=[]
    for classifier in label:
      n.append(score_doc_label_2(document, classifier, classifyList))
    return label[n.index(max(n))]   
# classify documents by the first smoothing method
def classify_documents(docs, classifyList):
    return [classify_nb(document,classifyList) for document in docs]
# classify documents by the second smoothing method  
def classify_documents_2(docs, classifyList):
    return [classify_nb_2(document,classifyList) for document in docs]   
#accuracy calculation (The number of correct predicted / the total number of samples)   
def accuracy(true_labels, guessed_labels):
    n=0
    b=9531
    total=len(true_labels)
    print(total)
    for i,j in zip(true_labels, guessed_labels):
        if i == j:
            n+=1 
        print(b)
        print(i,j)
        b+=1
    return float(n)/total 

#To calculate the credible interval using Bayesian approach with a Beta distribution. 
  
def interval_estimate(total,accuracy):
        Nright=total*accuracy
        #Assume it is a symmetric distribution with non specific prior a=1 and b=1.
        Nwrong=total-Nright
        a=1
        b=1
        p_mle=accuracy
        posterior = scipy.stats.beta(a + Nright, b + Nwrong)
        p_range = np.arange(0, 1, 0.001)
        plt.plot(p_range, posterior.pdf(p_range));
        ci_95 = posterior.interval(0.95)
        plt.plot([ci_95[0], ci_95[0]], [0, posterior.pdf(ci_95[0])], 'r')
        plt.plot([ci_95[1], ci_95[1]], [0, posterior.pdf(ci_95[1])], 'r')
        plt.plot(p_mle,posterior.pdf(p_mle),'o')
        # fill the area under the curve with light blue
        between = np.arange(ci_95[0], ci_95[1]+0.01, 0.01)
        plt.fill_between(between, posterior.pdf(between), color='LightBlue');
        print(f'ML / MAP estimate: {p_mle:.2}')
        print(f'95% credible interval: {ci_95[0]:.2} {ci_95[1]:.2}')
# cross-validation, the data is divided into N parts of equal size. 
#Each part once becomes a test set, then the other parts form the training set. 
def crossValidation(N,all_docs,all_labels):
    total_guessed_labels=[]
    for fold_nbr in range(N):
        split_point_1 = int(float(fold_nbr)/N*len(all_docs))
        split_point_2 = int(float(fold_nbr+1)/N*len(all_docs))
        train_docs_fold = all_docs[:split_point_1] + all_docs[split_point_2:]
        train_labels_fold = all_labels[:split_point_1] + all_labels[split_point_2:]
        eval_docs_fold = all_docs[split_point_1:split_point_2]
        trained_classifier=train_nb(train_docs_fold, train_labels_fold)
        guessed_labels=classify_documents(eval_docs_fold, trained_classifier)
        total_guessed_labels+=guessed_labels        
    total=len(all_labels)
    ac=accuracy(all_labels, total_guessed_labels)
    interval_estimate(total,ac)
        

all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')
split_point = int(0.80*len(all_docs))
print(split_point)
print(len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]  
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

#Sanity Check 1
print(np.exp(score_doc_label(['great'],'pos',train_nb(train_docs, train_labels))))
print(np.exp(score_doc_label(['great'],'neg',train_nb(train_docs, train_labels))))
print(np.exp(score_doc_label(['bad'],'pos',train_nb(train_docs, train_labels))))
print(np.exp(score_doc_label(['bad'],'neg',train_nb(train_docs, train_labels))))
#Sanity Check2
print(np.exp(score_doc_label(['a', 'top-quality', 'performance'],'pos',train_nb(train_docs, train_labels))))

#accurcy calculation
guessed_labels=classify_documents(eval_docs, train_nb(train_docs, train_labels))
print(accuracy(eval_labels, guessed_labels))

#Computing an interval estimate for the accuracy using the Bayesian method
interval_estimate(2383,0.81)

#p-value calculation
print(scipy.stats.binom_test(1936,2383,0.8))

#second classifier
guessed_labels_2=classify_documents_2(eval_docs, train_nb(train_docs, train_labels))
#Comparing two classifiers using McNemar test
allTrue=0
oneTrue=0
twoTrue=0
allFalse=0
for a,b,c in zip(guessed_labels,guessed_labels_2,eval_labels):
    if a == c and b==c:
        allTrue+=1 
    elif a==c and b!=c:
        oneTrue+=1
    elif a!=c and b==c:
        twoTrue+=1
    elif a!=c and b!=c:
        allFalse+=1
print(allTrue,  oneTrue, twoTrue, allFalse)
#p-value calculation for the second smoothy method
print(scipy.stats.binom_test(twoTrue+allTrue,2383,0.8))

#Computing an interval estimate for the accuracy using the Bayesian method based on cross-validation
crossValidation(10,all_docs, all_labels)

print(guessed_labels)
print(eval_labels)

  
    

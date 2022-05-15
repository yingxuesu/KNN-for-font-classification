#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 20:05:12 2021

@author: yingxue
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

###load the dataset
data1=pd.read_csv('/home/yingxue/Desktop/AGENCY.csv')
Font1=data1.drop(columns=['fontVariant','m_label','orientation','m_top','m_left','originalH','originalW','h','w'])

data2=pd.read_csv('/home/yingxue/Desktop/BAITI.csv')
Font2=data2.drop(columns=['fontVariant','m_label','orientation','m_top','m_left','originalH','originalW','h','w'])

data3=pd.read_csv('/home/yingxue/Desktop/BELL.csv')
Font3=data3.drop(columns=['fontVariant','m_label','orientation','m_top','m_left','originalH','originalW','h','w'])

data4=pd.read_csv('/home/yingxue/Desktop/BAITI.csv')
Font4=data4.drop(columns=['fontVariant','m_label','orientation','m_top','m_left','originalH','originalW','h','w'])

CL1=Font1.loc[(Font1['strength'] == 0.4) & (Font1['italic'] == 0)]
CL2=Font2.loc[(Font2['strength'] == 0.4) & (Font2['italic'] == 0)]
CL3=Font3.loc[(Font3['strength'] == 0.4) & (Font3['italic'] == 0)]
CL4=Font4.loc[(Font4['strength'] == 0.4) & (Font4['italic'] == 0)]

DATA=pd.concat([CL1,CL2,CL3,CL4])
DATA=DATA.drop(columns=['font','strength','italic'])
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(DATA)
SDATA = pd.DataFrame(scaler.transform(DATA))

true=[1]*251+[2]*412+[3]*239+[4]*412

from sklearn.model_selection import train_test_split
###split training and test set on CLi
trainCL1,testCL1,trainY1,testY1=train_test_split(CL1.drop(columns=['font','strength','italic']),[1]*251,test_size=0.2)
trainCL2,testCL2,trainY2,testY2=train_test_split(CL2.drop(columns=['font','strength','italic']),[2]*412,test_size=0.2)
trainCL3,testCL3,trainY3,testY3=train_test_split(CL3.drop(columns=['font','strength','italic']),[3]*239,test_size=0.2)
trainCL4,testCL4,trainY4,testY4=train_test_split(CL4.drop(columns=['font','strength','italic']),[4]*412,test_size=0.2)

### Generate the whole training set Train and the whole test set Test
### and standarize the dataset
Train = pd.concat([trainCL1,trainCL2,trainCL3,trainCL4])
scaler = preprocessing.StandardScaler().fit(Train)
Train = pd.DataFrame(scaler.transform(Train))
TrainY=pd.DataFrame(trainY1+trainY2+trainY3+trainY4)
Test = pd.concat([testCL1,testCL2,testCL3,testCL4])
scaler = preprocessing.StandardScaler().fit(Test)
Test = pd.DataFrame(scaler.transform(Test))
TestY=pd.DataFrame(testY1+testY2+testY3+testY4)

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import math
TraAcc=[]## store the accuracy on training set for different k in this list
TestAcc=[]## store the accuracy on test set for different k in this list
margin=[]## store the margin on test set in this list
for k in [5,10,15,20,30,40,50]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(Train,TrainY)
    y_train_pred= model.predict(Train) 
    y_test_pred=model.predict(Test)
    acc1=metrics.accuracy_score(TrainY, y_train_pred)
    acc2=metrics.accuracy_score(TestY, y_test_pred)
    margin.append(math.sqrt(acc2*(1-acc2)/265))
    TraAcc.append(acc1)
    TestAcc.append(acc2)
l=np.asarray([5,10,15,20,30,40,50])
###Compute the 90% confidence interval [CI1 CI2]
CI1=TestAcc-1.6*np.asarray(margin)
CI2=TestAcc+1.6*np.asarray(margin)

###plot the accuracy curve on training set(red), test set(black) and the 90% confidence 
###interval of accuracy on test set(yellow area) in the same figure
plt.fill_between(l,CI1,CI2,color='yellow',label='90% CI')
plt.plot(l,TraAcc,'*--',color='red',label='TrainAccuracy')
plt.plot(l,TestAcc,'.--',color='black',label='TestAccuracy')
plt.legend()
### the interval [a b] is [25 35]

TraAcc=[]## store the accuracy on training set for different k in this list
TestAcc=[]## store the accuracy on test set for different k in this list
margin=[]## store the margin on test set in this list
for k in range(25,35):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(Train,TrainY)
    y_train_pred= model.predict(Train) 
    y_test_pred=model.predict(Test)
    acc1=metrics.accuracy_score(TrainY, y_train_pred)
    acc2=metrics.accuracy_score(TestY, y_test_pred)
    margin.append(math.sqrt(acc2*(1-acc2)/265))
    TraAcc.append(acc1)
    TestAcc.append(acc2)
l=range(25,35)
###Compute the 90% confidence interval [CI1 CI2]
CI1=TestAcc-1.6*np.asarray(margin)
CI2=TestAcc+1.6*np.asarray(margin)

###plot the accuracy curve on training set(red), test set(black) and the 90% confidence 
###interval of accuracy on test set(yellow area) in the same figure
plt.fill_between(l,CI1,CI2,color='yellow',label='90% CI')
plt.plot(l,TraAcc,'*--',color='red',label='TrainAccuracy')
plt.plot(l,TestAcc,'.--',color='black',label='TestAccuracy')
plt.legend()
### the optimal k* is 30


from sklearn.metrics import confusion_matrix
model = KNeighborsClassifier(n_neighbors=30)
model.fit(Train,TrainY)
y_train_pred= model.predict(Train) 
y_test_pred=model.predict(Test)
confusion_matrix(TrainY, y_train_pred)
confusion_matrix(TestY, y_test_pred)

l21=[]#l21 is the index of cases whose true class is 2 but are misclassified to class 1
l23=[]
l24=[]
trainy=TrainY.to_numpy()
for i in range(1049):
    if trainy[i]==2 and y_train_pred[i]==1:
        l21.append(i)
    if trainy[i]==2 and y_train_pred[i]==3:
        l23.append(i)
    if trainy[i]==2 and y_train_pred[i]==4:
        l24.append(i)

#### pick misclassified case case1  
tt=Train.to_numpy()
case1=tt[l21[0]]
dis1=[]### compute the distances between cases in the training set and case1
for i in range(1049):
    dis1.append(np.linalg.norm(tt[i]-case1))

id1=np.argsort(dis1) ###sort the dis1 and find the 15 cases with 15 smallest distance
### and print their true class
for i in range(15):
    print(TrainY.loc[id1[i]])

### pick misclassified case case 2
case2=tt[l23[2]]
dis2=[]### compute the distances between cases in the training set and case2
for i in range(1049):
    dis2.append(np.linalg.norm(tt[i]-case2))

id2=np.argsort(dis2)###sort the dis2 and find the 15 cases with 15 smallest distance
### and print their true class
for i in range(15):
    print(TrainY.loc[id2[i]])
    
case3=tt[l24[2]]
dis3=[]### compute the distances between cases in the training set and case3
for i in range(1049):
    dis3.append(np.linalg.norm(tt[i]-case3))
id3=np.argsort(dis3)###sort the dis3 and find the 15 cases with 15 smallest distance
### and print their true class
for i in range(15):
    print(TrainY.loc[id3[i]])
       




### compute the correlation matrix      
CORR=SDATA.corr()
###Eigenvalues Eigen vector of CORR
from numpy import linalg as LA     
l, w = LA.eig(CORR) #l is eigenvalues, w is the eigen vector

L=np.sort(l)[::-1] #sort the eigenvalues in decreasing order
idx = np.argsort(-l) # index of sorted eigenvalues
plt.plot(L)
plt.title('eigenvalues')

###find r such that PEV(r)>90%
for r in range(400):
    if sum(L[0:r])>0.9*400:
        print(r)

W=w[:,idx[0:60]] #find eigenvectors that corresponding to the largest r eigenvalues
sdata=SDATA.to_numpy()
Z=np.matmul(sdata,W)#transform the data set using PCA analysis
ZDATA=pd.DataFrame(Z)

### Compute the training set and test set after PCA analysis
NTrain=pd.DataFrame(np.matmul(Train.to_numpy(),W))
NTest=pd.DataFrame(np.matmul(Test.to_numpy(),W))
from sklearn.metrics import confusion_matrix
model = KNeighborsClassifier(n_neighbors=30)
model.fit(NTrain,TrainY)
y_train_pred= model.predict(NTrain) 
y_test_pred=model.predict(NTest)
confusion_matrix(TrainY, y_train_pred)
confusion_matrix(TestY, y_test_pred)
metrics.accuracy_score(TrainY, y_train_pred)
metrics.accuracy_score(TestY, y_test_pred)    

l2=[]#l2 is the index of cases that belongs to class 2
l1=[]
l3=[]
l4=[]
for i in range(1049):
    if trainy[i]==2:
        l2.append(i)
    if trainy[i]==1:
        l1.append(i)
    if trainy[i]==3:
        l3.append(i)
    if trainy[i]==4:
        l4.append(i)


CL2DATA=NTrain.iloc[l2]
CL1DATA=NTrain.iloc[l1]
CL3DATA=NTrain.iloc[l3]
CL4DATA=NTrain.iloc[l4]
####plot the class1 and class 2 cases after PCA analysis
plt.scatter(CL2DATA[0], CL2DATA[1],  s=5,color='black', label='class2')
plt.scatter(CL1DATA[0], CL1DATA[1],  s=5,color='red', label='class1')
plt.title('class 1 and class 2')
plt.legend()
####plot the class2 and class3 cases after PCA analysis
plt.scatter(CL2DATA[0], CL2DATA[1],  s=5,color='black', label='class2')
plt.scatter(CL3DATA[0], CL3DATA[1],  s=5,color='green', label='class3')
plt.title('class 2 and class 3')
plt.legend()


####plot the class3 and class 4 cases after PCA analysis
plt.scatter(CL4DATA[0], CL4DATA[1],  s=5,color='yellow', label='class4')
plt.scatter(CL2DATA[0], CL2DATA[1],  s=10, color='black', label='class2')
plt.title('class 2 and class 4')
plt.legend()

###plot class 2 cases and misclassified cases
ERR21=NTrain.iloc[l21]
ERR23=NTrain.iloc[l23]
ERR24=NTrain.iloc[l24]
plt.scatter(CL2DATA[0], CL2DATA[1],  s=10,color='black', label='class2')
plt.scatter(ERR23[0], ERR23[1],  s=10,color='green', label='misclass 3')
plt.scatter(ERR21[0], ERR21[1],  s=10,color='red', label='misclass 1')
plt.scatter(ERR24[0], ERR24[1],  s=10,color='yellow', label='misclass 4')
plt.legend()










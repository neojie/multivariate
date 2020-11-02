#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 12:45:02 2020
source https://datascienceplus.com/understanding-the-covariance-matrix/

https://meshlogic.github.io/posts/jupyter/linear-algebra/linear-algebra-numpy-2/
https://github.com/neojie/PythonDataScienceHandbook/blob/master/notebooks/05.09-Principal-Component-Analysis.ipynb
https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis
https://towardsdatascience.com/pca-clearly-explained-how-when-why-to-use-it-and-feature-importance-a-guide-in-python-7c274582c37e
@author: jiedeng
"""
"""
variance vs. covariance
correlation matrix = covriance/xx
PCA find features that are important in the reduced dataset
application
"""
import pandas as pd

from Data import DataSets
import numpy as np
from MDAnalysis.lib import distances
import matplotlib.pyplot as plt
import copy



## 1. variance vs. covriance 
# variance for an array vs. covariance for two arrays
x = np.array([.3,.3,.4])
y = np.array([0,.8,.2])
np.cov(x,y)


# Calculate covariance matrix 

# Covariance
def cov(x, y):
    xbar, ybar = x.mean(), y.mean()
    return np.sum((x - xbar)*(y - ybar))/(len(x) - 1)

# method 1 Covariance matrix
def cov_mat(X):
    return np.array([[cov(X[0], X[0]), cov(X[0], X[1])], \
                     [cov(X[1], X[0]), cov(X[1], X[1])]])

X =  np.stack((x,y),axis=0)

cov_mat(X) # (or with np.cov(X.T))

# mehtod #2
mu_X = X.mean(axis=1).reshape(2,1)
np.matmul((X - mu_X),(X.T - mu_X.T))/2

# bendmark
covariance = np.cov(X)

## 2. covariance to correlation matrix 
def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation

correlation_from_covariance(np.cov(X))

## 3. PCA find the best 

def get_train_data(data):
    """
    data: DataSets instance
    ------
    DataSets methods can only load each set directory seperately
    This is a wrapper to concatenate all training data together
    """
    train_dirs = data.train_dirs
    data.load_batch_set(train_dirs[0]);  out = data.batch_set
    
    if len(train_dirs)>1:
        for i in range(1,len(train_dirs)):
            data.load_batch_set(train_dirs[i]); add  = data.batch_set
            for key in out:
                out[key] = np.concatenate((out[key],add[key]))
    return out

out_pd = pd.read_excel('/Users/jiedeng/GD/ppt/2020/extreme_filter4_with_state.xlsx')

deepmds = out_pd['local_path'].values


i =8

data = DataSets(deepmds[i], 'set', shuffle_test = False)
train_data = get_train_data (data)  

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
projected = pca.fit_transform(train_data['coord'])

print(projected.shape)

plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();

plt.scatter(projected[:, 0], projected[:, 1])


#Choosing the number of components
pca = PCA().fit(train_data['coord'])
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');






#X = np.stack((x-0.333, y-0.333), axis=0)
#np.cov(X)
#np.dot(X,X.T)
#np.cov(np.stack((x, y), axis=0))
#
#np.sum(X[0]*X[1]/(len(X[0])-1))


np.cov(train_data['coord'][0],train_data['coord'][1])

np.cov(train_data['coord'])

w,v = np.linalg.eig(np.cov(train_data['coord'][0],train_data['coord'][1]))


###
A = np.array([[2, 0, 0], [0 ,3 ,4], [0, 4 ,9]])
A = np.array([[4, 1, 2], [1 ,9 ,-3], [2, -3 ,25]])
lam, U = np.linalg.eig(A)

U*np.diag(lam)*U.T

A = np.matrix('2 0 0; 0 3 4; 0 4 9')
lam, U = np.linalg.eig(A)
U*np.diag(lam)*U.T

import numpy as np




## PCA

from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

pca = PCA().fit(digits.data[:63])
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

pca = PCA().fit(digits.data[:63])



### extract the most important component
# https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis
train_data['coord'].T.shape

train_data['coord'].T

pca = PCA().fit(train_data['coord'].T)

# we have 480 samples with 50 features, here 480 is 160*3
variance_ratio = np.cumsum(pca.explained_variance_ratio_)
plt.plot(variance_ratio)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

pca = PCA(5).fit(train_data['coord'].T)
pca.explained_variance_ratio_

print(abs( pca.components_ ))

threshold = .99
len(variance_ratio[variance_ratio<.99])/len(variance_ratio)
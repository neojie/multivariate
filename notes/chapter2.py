#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 12:45:02 2020

variance vs. covariance
correlation matrix = covriance/xx

source 
https://datascienceplus.com/understanding-the-covariance-matrix/
https://meshlogic.github.io/posts/jupyter/linear-algebra/linear-algebra-numpy-2/

see also notebook/math/pca
@author: jiedeng
"""

import numpy as np



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

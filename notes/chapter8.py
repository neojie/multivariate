#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
see also chapter 2.py
variance vs. covariance
correlation matrix = covriance/xx
PCA find features that are important in the reduced dataset
application
focus on PCA and its application

https://github.com/neojie/PythonDataScienceHandbook/blob/master/notebooks/05.09-Principal-Component-Analysis.ipynb
https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis
https://towardsdatascience.com/pca-clearly-explained-how-when-why-to-use-it-and-feature-importance-a-guide-in-python-7c274582c37e
https://stackoverflow.com/questions/4383571/importing-files-from-different-folder?page=1&tab=votes#tab-top
@author: jiedeng
"""
import numpy as np
"""
# example 8.1
# to get a more specific example, I want first construct a matrix based on the given covariance matrix
# I solve 
L*L.T/(n-1) = Cov
L = cholesky(Cov)*(n-1)
Note L = X - mu(X)
L is not the only one that gives Cov, bc/ to generate n*n cov, we need n col vector, but the length of colv vector is unknown
the reference below is more detailed disucssion on how to get n*m data, rather than n
ref 
https://math.stackexchange.com/questions/17575/going-back-from-a-correlation-matrix-to-the-original-matrix
"""
# exchangestack example

cov = np.array([[1.00  ,     -0.38    ,    0.22  ,     -0.85],
                [-0.38    ,    1.00    ,    0.30     ,   0.57],
                [0.22   ,     0.30    ,    1.00     ,   0.27],
                [-0.85    ,    0.57     ,   0.27   ,     1.00]])

L=np.linalg.cholesky(cov*3)
np.matmul(L,L.T)/3


# example 9.1
cov = np.array([[1,-2,0],[-2,5,0],[0,0,2]])
L=np.linalg.cholesky(cov*2)
np.matmul(L,L.T)/2

#### try the stack method to get the original matrix, not working
mu = L.mean(axis=1).reshape(3,1)
np.matmul(L - mu,(L-mu).T)/2

v = 3
n = 10
F = np.random.rand(v ,n)
covF = np.matmul(F,F.T)/(3-1)
choF = np.linalg.cholesky(covF)

L = np.linalg.cholesky(cov)

dat = np.matmul(np.matmul(L,choF),F)

sdevs = np.diag([1,2,3]) 
means = np.array([20,10,40])
simuldata = np.matmul(sdevs,dat) + means.reshape(3,1)

np.cov(simuldata)
np.cov(dat)
#### try the stack method to get the original matrix

"""
array([[ 1.41421356,  0.        ,  0.        ],
       [-2.82842712,  1.41421356,  0.        ],
       [ 0.        ,  0.        ,  2.        ]])

X1 = [ 1.41421356,  0.        ,  0.        ]
X2 = [-2.82842712,  1.41421356,  0.        ]
X3 = [ 0.        ,  0.        ,  2.        ]
"""


# solve for 
lam,v = np.linalg.eig(cov)
np.matmul(np.matmul(v,np.diag(lam)),v.T)## eigen decomposition or spectral decompositin

# Note v*np.diag(lam)*v.T gives the incorrect results
"""
lam[0] -> 5.82842712; v[0]

array([0.17157288, 5.82842712, 2.        ])

array([[-0.92387953,  0.38268343,  0.        ],
       [-0.38268343, -0.92387953,  0.        ],
       [ 0.        ,  0.        ,  1.        ]])

sort lam as descending
Y1 corresonds to max(lam)

"""
X1,X2,X3 = L
lam3,lam1,lam2 =lam
v3,v1,v2 = v
Y1 = np.matmul(v1,L)
#cf.
v1[0]*X1 + v1[1]*X2 + v1[2]*X3
np.var(Y1)
np.var(X1)




Y1 = np.matmul(v1,L.T)
Y2 = np.matmul(v2,L.T)
Y3 = np.matmul(v3,L.T)


np.var(Y1)
np.var(Y1)
np.var(Y1)
### not really working because X is not corret
###
"""
https://towardsdatascience.com/pca-clearly-explained-how-when-why-to-use-it-and-feature-importance-a-guide-in-python-7c274582c37e
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
plt.style.use('ggplot')
# Load the data
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Z-score the features
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
# The PCA model
pca = PCA(n_components=2) # estimate only 2 PCs
X_new = pca.fit_transform(X) # project the original data into the PCA space


fig, axes = plt.subplots(1,2)
axes[0].scatter(X[:,0], X[:,1], c=y)
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('Before PCA')
axes[1].scatter(X_new[:,0], X_new[:,1], c=y)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('After PCA')
plt.show()

fig, axes = plt.subplots(1,2)
axes[0].scatter(X[:,2], X[:,3], c=y)
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[0].set_title('Before PCA')
axes[1].scatter(X_new[:,0], X_new[:,1], c=y)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('After PCA')
plt.show()




### show eigen of Cov is the variance of Y

pca.explained_variance_ # the eigen of Cov
pca.explained_variance_ratio_  # ratio
pca.components_ # Y1 = a1X1 + a2X2+  .. , [a1,a2,...], higher coef, higher weight



### biplot  part not studied yet



"""
another example


from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape

pca = PCA().fit(digits.data[:63])
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

pca = PCA().fit(digits.data[:63])

"""
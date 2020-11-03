"""
/Users/jiedeng/Documents/ml/deepmd-kit/my_example/codes/tmp.py
"""
import pandas as pd
import numpy as np
from MDAnalysis.lib import distances
import matplotlib.pyplot as plt
import copy
import sys
sys.path.append('/Users/jiedeng/Documents/ml/deepmd-kit/my_example/codes/Data.py')
from Data import DataSets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


import dpdata
vasp_multi_systems = dpdata.MultiSystems.from_dir(dir_name='/Users/jiedeng/Documents/tmp/jd848/project_folder/pv+hf/3k/solid1/r3-3k/', file_name='OUTCAR', fmt='vasp/outcar')
from dpdata import LabeledSystem
ls = LabeledSystem('/Users/jiedeng/Documents/tmp/jd848/project_folder/pv+hf/3k/solid1/r3-3k/OUTCAR',fmt='outcar')
print(ls.data['coords'].shape) # (5000, 160, 3)


scaler = StandardScaler()


"""
we get n sample, we need find the most important or representative coord so that 
them can represent the rest of features.
1) z-score variables
2) eigendecomposition of covariance matrix, covriance matrix should be n*n, not 480*480
3) sort eigenvalues 
3) projection of the original normalized data onto the PCA space
If we use above protocal, dat should be dat
But if we use PCA module directly, input should be dat.T
"""
### benchmark
n = 5 
# we get n sample, but here we need treat them as feature, we calculate the 
dat = ls.data['coords'][:n,:,:]
dat = dat.reshape((n,480))
scaler.fit(dat)
dat = scaler.transform(dat)

cov = np.cov(dat)
np.linalg.eig(cov)# array([ 4.68590527e+00,  3.14826924e-01, -5.01335085e-16,  4.15060238e-03, 2.33741058e-03]),

pca = PCA(2).fit(dat.T)
#pca.components_
pca.explained_variance_  # array([4.68590527, 0.31482692]), the largest two eigenvalues



##

n = 100
# we get n sample, but here we need treat them as feature, we calculate the 
dat = ls.data['coords'][:n,:,:]
dat = dat.reshape((n,480))
scaler.fit(dat)
dat = scaler.transform(dat)

pca = PCA().fit(dat.T)
variance_ratio = np.cumsum(pca.explained_variance_ratio_)
plt.figure()
plt.plot(variance_ratio)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

threshold = .99
len(variance_ratio[variance_ratio<.99])/len(variance_ratio)

plt.plot(abs(pca.components_[7,:]),'.')
plt.plot(abs(pca.components_[8,:]),'.')
tmpx = abs(pca.components_[8,:])
len(tmpx[tmpx>0.1])
plt.plot([])

plt.plot(abs(pca.components_[90,:]),'.')
plt.plot(abs(pca.components_[91,:]),'.')

tmpx = abs(pca.components_[91,:])
len(tmpx[tmpx>0.1])




############################################

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

out_pd  = pd.read_excel('/Users/jiedeng/GD/ppt/2020/extreme_filter4_with_state.xlsx')
deepmds = out_pd['local_path'].values

i = 8

data       = DataSets(deepmds[i], 'set', shuffle_test = False)
train_data = get_train_data (data)  


pca       = PCA(n_components=2)
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


## PCA

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




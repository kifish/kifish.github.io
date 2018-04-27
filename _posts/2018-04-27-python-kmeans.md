---
layout: post
tags: [python,ml,他山之石]
---

http://blog.csdn.net/github_36326955/article/details/54999627
```
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import sklearn
X,y = make_blobs(n_samples=1000,n_features=2,centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=[0.4,0.2,0.2,0.2]
                 ,random_state=9)
plt.scatter(X[:,0],X[:,1],marker='o')
plt.show()

from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=2,random_state=9).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()

from sklearn import metrics
print(metrics.calinski_harabaz_score(X,y_pred))

y_pred = KMeans(n_clusters=3,random_state=9).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()
print(metrics.calinski_harabaz_score(X,y_pred))

y_pred = KMeans(n_clusters=4,random_state=9).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()
print(metrics.calinski_harabaz_score(X,y_pred))

from sklearn.cluster import MiniBatchKMeans
#MiniBatchKMeans
for index,val in enumerate((2,3,4,5)):
    plt.subplot(2,2,index+1)
    y_pred = MiniBatchKMeans(n_clusters=val, batch_size=200,random_state=9).fit_predict(X)
    score = metrics.calinski_harabaz_score(X,y_pred)
    plt.scatter(X[:,0],X[:,1],c=y_pred)
    plt.text(0.99,0.01,('val=%d, score:%.2f'%(val,score)),transform=plt.gca().transAxes,size=10,
             horizontalalignment='right')

plt.show()
```
python代码实现可参考：

http://blog.csdn.net/dream_angel_z/article/details/46343597

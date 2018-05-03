---
layout: post
tags: [python,ml,他山之石]
published : true
---

#http://blog.csdn.net/u010099080/article/details/53560426

mpl_toolkits.mplot3d 这个包在matplotlib里面
python mpl_toolkits installation issue：

https://stackoverflow.com/questions/37661119/python-mpl-toolkits-installation-issue

```python
# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn import  decomposition
import  matplotlib.pyplot as plt
import  numpy as np
import  seaborn

from mpl_toolkits.mplot3d import Axes3D

mnist = datasets.load_digits()
X = mnist.data
y = mnist.target

pca = decomposition.PCA(n_components=3)
new_X = pca.fit_transform(X)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(new_X[:,0],new_X[:,1],new_X[:,2],c=y,cmap=plt.cm.spectral)
plt.show()
```

python3D柱状图：

https://www.jianshu.com/p/bb8b25096df4

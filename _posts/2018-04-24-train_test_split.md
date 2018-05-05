---
layout: post
tags: [sklearn,ml]
published : false
---
stratify是为了保持split前类的分布。比如有100个数据，80个属于A类，20个属于B类。如果train_test_split(X,y, test_size=0.25, stratify = y), 那么split之后数据如下：
training: 75个数据，其中60个属于A类，15个属于B类。
testing: 25个数据，其中20个属于A类，5个属于B类。
用了stratify参数，training集和testing集的类的比例是 A：B= 4：1，等同于split前的比例（80：20）。通常在这种类别分布不平衡的情况下会用到stratify。
-https://blog.csdn.net/heifan2014/article/details/79040744  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify = y,shuffle = True)
实际上如果类别分布很不平衡，极端情况如某一类只有一个样本，那么train_test_split很大可能会报错，因为没法保证training set和test set比例一致，并且这时候还有test_size=0.2的要求。如果适当减小test_size，出错的可能会小一点。   


/Users/k/anaconda3/lib/python3.6/site-packages/sklearn/utils/validation.py
```python
def _num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)

```

最后发现不是split的问题
如果传进去的y是np.array('0006159194')就会报错。
至于y为什么会变成形如np.array('0006159194')，遍历的时候变量名冲突了
见2018-04-25-字典序.md

```python
import numpy as np
def _num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)

#x = [[1]]   #1
#x = [1]     #1
#x = ['123','123'] #2
'''  7
x = ['0006161207','0006166482','0006161207','0006166482','0006161207',
 '0006161207','0006161207']
'''

#x = np.array([['1','123','123']])   1
#x = np.array(['1','123','123'])   3
#x = '123'   #3
#x = '2324'  #4
#x = 1234    # TypeError: Expected sequence or array-like, got <class 'int'>
#x = []  #0
#x = np.array('123')  #TypeError: Singleton array array('123', dtype='<U3') cannot be considered a valid collection.
'''
x = np.array('0006159194')#TypeError: Singleton array array('0006159194', dtype='<U10') cannot be considered a valid collection.
print(len(x))  #TypeError: len() of unsized object
'''
#x = np.array(['0006159194'])  #1
#print(np.array('0006159194')[0]) IndexError: too many indices for array
#print(type(np.array('0006159194'))) #<class 'numpy.ndarray'>

x = ['0006159194'] #1
print(_num_samples(x))




```



去掉也没关系,即使0.2不够也会自动取整，但是以下这段代码会提高acc等指标，因为对于小样本集的情况，只用了一个测试样本
```python

class2sample_num = {}
for item in y:
    if item in class2sample_num:
        class2sample_num[item] += 1
    else:
        class2sample_num[item] = 1
min = 1000000
for k,v in class2sample_num.items():
    if v < min:
        min = v
flag = False
if min <= 5:
    flag = True
while True:
    '''
    if flag:
        print("minnnnnnnnnn")
        print(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1,shuffle = True)  #随机抽取
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle = True)
```

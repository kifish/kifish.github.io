---
layout: post
tags: [python,ml,他山之石]
---

见http://blog.csdn.net/zouxy09/article/details/20319673：   
getA()函数是在numpy模块中的，功能是将矩阵转化成数组

主要bug原因是：有的地方把python当matlab了，比如对matrix的索引，两者其实有一些区别的。

原文中的代码可能随机化的地方有bug。

```python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
import random
#sigmoid
def sigmoid(z):
    return 1.0/(1+np.exp(-z))

#opts = {alpha:a_val,max_epoch_times:m_val}
def lr_train(X,Y,opts):
    start = time.time()
    m,n = np.shape(X)
    # X = [np.ones(m,1),X] if matlab
    X = np.concatenate((np.ones((m,1)),X),axis=1)

    n += 1
    alpha = opts['alpha']
    max_epoch_times = opts['max_epoch_times']
    w = np.ones((1,n))
    w = np.mat(w)
    for epoch in range(max_epoch_times):
        if opts['optimizer'] == 'GradientDescent':
            pred = sigmoid(X*np.transpose(w))
            err = Y-pred
            #maxmize the likelihood
            w += alpha*np.transpose(err)*X

        elif opts['optimizer'] == 'StoGradDescent':
            temp_m_range = range(m)
            temp_m_range = list(temp_m_range)
            #we can also use permutation to simplify
            #actually the result is equal to direct loop
            for iter in range(m):
                sample_idx = int(random.uniform(0,len(temp_m_range)))
                pred = sigmoid(X[sample_idx,:]*np.transpose(w))
                err = Y[sample_idx]-pred
                w += alpha*err*X[sample_idx,:]
                del_idx = 0
                for idx,val in enumerate(temp_m_range):
                    if val == sample_idx:
                        del_idx = idx
                        break
                del temp_m_range[del_idx]

        #decrease alpha with iteration
        elif opts['optimizer'] == 'SmoothStoGradDescent':
            temp_m_range = range(m)
            temp_m_range = list(temp_m_range)
            #we can also use permutation to simplify
            for iter in range(m):
                alpha = 4.0 / (1.0 + epoch + iter) + 0.01
                sample_idx = int(random.uniform(0, len(temp_m_range)))
                pred = sigmoid(X[sample_idx,:] * np.transpose(w))
                err = Y[sample_idx] - pred
                w += alpha * err * X[sample_idx,:]
                #del temp_m_range[sample_idx] this code means one sample maybe selected for some times
                """
                The code below means one sample should not be selected for over one time.
                However, the code above can bring better effects which confuses me.
                Maybe SmoothStoGradDescent is deeply affected by randomness.
                """
                del_idx = 0
                for idx,val in enumerate(temp_m_range):
                    if val == sample_idx:
                        del_idx = idx
                        break
                del temp_m_range[del_idx]
        else:
            raise NameError('Invalid optimizer type!')

        print("Training is completed. It took %ss" %(time.time()-start))
        return w

def lr_test(X,Y,w):
    m,n = np.shape(X)
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    acc_cnt = 0
    for idx in range(m):
        pred = 0
        if sigmoid(X[idx,:] * np.transpose(w)) > 0.5:
            pred = 1
        if Y[idx] == pred:
            acc_cnt += 1
        acc = float(acc_cnt)/m
    return acc

def lr_plot(X,Y,w):
    m,n = np.shape(X)
    X= np.concatenate((np.ones((m, 1)), X), axis=1)
    n += 1
    #print(Y)
    #print("n = %d"%n)
    try:
        if n ==3 :
            for i in range(m):
                #note: here numpy is different from matlab
                #Y[i] != Y[i,0] because type(Y) is np.matrix
                if int(Y[i,0]) == 0:
                    plt.plot(X[i, 1], X[i, 2], 'ob')

                elif int(Y[i,0]) == 1:
                    plt.plot(X[i, 1], X[i, 2], 'or')

    except Exception as e:
        print("The number of features in invalid")

    #X is np.mat
    min_x = min(X[:,1])[0,0]
    max_x = max(X[:,1])[0,0]
    """
    print(w)
    print((type(w)))
    """
    w_array = w.getA() # conver mat to array
    """
    print(w_array)
    print(type(w_array))
    """
    y_min_x = float(-w_array[0][0]-w_array[0][1]*min_x)/w_array[0][2]
    y_max_x = float(-w_array[0][0]-w_array[0][1]*max_x)/w_array[0][2]
    plt.plot([min_x,max_x],[y_min_x,y_max_x],'-g')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def load_data():
    train_x = []
    train_y = []
    f = open('/Users/k/Downloads/data.txt')
    for line in f.readlines():
        line_arr = line.strip().split()
        train_x.append([float(line_arr[0]),float(line_arr[1])])
        train_y.append([float(line_arr[2])])
        #note:train_y.append([float(line_arr[2])]) should not be train_y.append(float(line_arr[2]))

    return np.mat(train_x),np.mat(train_y)

#load data
print("loading data")
train_x,train_y = load_data()
#maybe we should use other data to test
test_x = train_x
test_y = train_y

#train
print("training")
opts = {
    'alpha' :0.01,
    'max_epoch_times' : 100,
    'optimizer' : 'SmoothStoGradDescent'
    #'optimizer' : 'StoGradDescent'
    #'optimizer' : 'GradientDescent'
}

w = lr_train(train_x,train_y,opts)
#print(w)
#fit ;actually  we should use new data
acc = lr_test(test_x,test_y,w)

print("result:")
print("accuracy : %.3f%%" % (acc*100))
# (acc*100) should not be acc*100
lr_plot(train_x,train_y,w)

```

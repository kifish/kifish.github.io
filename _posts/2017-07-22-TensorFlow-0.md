Getting started with Machine Learning in 5 minutes
https://jizhi.im/blog/post/5min-ml  （可在网页上运行）


tf1.0

https://zhuanlan.zhihu.com/p/22410917

```python
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 2000
display_step = 50

# Training Data
train_X = numpy.asarray(
    [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = numpy.asarray(
    [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Create Model

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
activation = tf.add(tf.mul(X, W), b)

# Minimize the squared errors
cost = tf.reduce_sum(tf.pow(activation - Y, 2)) / (2 * n_samples)  # L2 loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # Gradient descent

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch + 1), "cost=", \
                "{:.9f}".format(sess.run(cost, feed_dict={X: train_X, Y: train_Y})), \
                "W=", sess.run(W), "b=", sess.run(b)

    print "Optimization Finished!"
    print "cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), \
        "W=", sess.run(W), "b=", sess.run(b)

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
```

可能会出现protobuf的问题

https://stackoverflow.com/questions/38680593/importerror-no-module-named-google-protobuf

注意：python3.x中range和xrange合并了。range不再是列表了，而是类似迭代器。


```python
import tensorflow as tf
# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()
# tf.global_variables_initializer 替代

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
```
 在高版本的tensorflow中是有MNIST的。




回归：


安装：http://www.cnblogs.com/shihuc/p/6593041.html
http://www.linuxidc.com/Linux/2017-03/142297.htm  （ubuntu下安装看这个比较好）
numpy 单独更新下。
sudo pip install numpy --upgrade
 
http://wiki.jikexueyuan.com/project/tensorflow-zh/get_started/os_setup.html（基于 VirtualEnv 的安装）
 ```
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random
hello = tf.constant("Hello, TensorFlow")
sess = tf.Session()
print(sess.run(hello))
a = tf.constant(2)
b = tf.constant(3)
with tf.Session() as sess:
    print("a=2,b=3")
    print("Addition with constants: %i" % sess.run(a+b))
    print("Multiplication with constants: %i" % sess.run(a*b))
#output
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a,b)
mul = tf.multiply(a,b)
with tf.Session() as sess:
    #run every operation with variable input
    print("Addition with varibales: %i" % sess.run(add,feed_dict={a:2,b:3}))
    print("Multiplication with variables: %i" %sess.run(mul,feed_dict={a:2,b:3}))
#output
 ```
 
 
 
 ```
import tensorflow as tf
import numpy as np
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.100, 0.200], x_data) + 0.300
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for step in xrange(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(b)
 线性回归：
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random
#Parameters
learning_rate = 0.01
training_epochs = 2000
display_step = 50
#Training Data
train_X = numpy.asarray(
    [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = numpy.asarray(
    [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = train_X.shape[0]
#tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")
#Create Model
#Set model weights
W = tf.Variable(rng.randn(),name="weight")
b = tf.Variable(rng.randn(),name="bias")
#construct a linear model
activation = tf.add(tf.multiply(X,W),b)
#Minimize the squared errors
cost = tf.reduce_sum(tf.pow(activation-Y,2)/(2*n_samples)) #L2 loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#Initialize the variables
init = tf.initialize_all_variables()
#Launch the graph
with tf.Session() as sess:
    sess.run(init)
#Fit all training data
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
#Display logs per epoch
        if epoch % display_step == 0:
            print("Epoch:",'%04d' %(epoch+1),"cost=",\
                  "{:.9f}".format(sess.run(cost,feed_dict={X:train_X,Y:train_Y})),\
                  "W=",sess.run(W),"b=",sess.run(b))
    print("Optimization Finished!")
    print("cost=",sess.run(cost,feed_dict={X:train_X,Y:train_Y}),\
          "W=",sess.run(W),"b=",sess.run(b))
#Graphic display
    plt.plot(train_X,train_Y,'ro',label='Original data')
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fitted line')
    plt.legend()
    plt.show()
 ```
tf onehot说明：
http://blog.csdn.net/wangbaoxing/article/details/79128668
 
#https://zhuanlan.zhihu.com/p/22410917
mnist数据加载的问题：由于网络原因，
 from tensorflow.examples.tutorials.mnist import input_data
 mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
这两行代码是无法下载到数据的。
可以手动下载（无需解压），然后放至对应目录。
例如：
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/usr/local/lib/python3.6/site-packages/tensorflow/examples/tutorials/mnist",one_hot=True)
/usr/local/lib/python3.6/site-packages/tensorflow/examples/tutorials/mnist   将数据放入该文件夹中

#mnist.py 会判断是否下载，因为源码里调用的函数是maybe_download(),这个函数会判断本地是否已经有文件了。
#https://www.cnblogs.com/huangshiyu13/p/6727745.html

```
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/usr/local/lib/python3.6/site-packages/tensorflow/examples/tutorials/mnist",one_hot=True)
#Parameters
learning_rate = 0.01
batch_size = 100
training_epochs = 25
display_step = 1
#Training Data
#tf Graph Input
X = tf.placeholder(tf.float32,[None,784])
#mnist data image if shape 28*28 = 784
y = tf.placeholder(tf.float32,[None,10])
#0-9 digits recognition => 10 classes
#Create Model
#Set model weights
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#construct a model
pred = tf.nn.softmax(tf.matmul(X,W)+b)
#Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1)) #L2 loss
#Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#Initialize the variables
init = tf.initialize_all_variables()
#Launch the graph
with tf.Session() as sess:
    sess.run(init)
#Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        #Loop over all batches
        for batch_idx in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            #Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X:batch_xs, y:batch_ys})
#Compute average loss
            avg_cost += c/total_batch
        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:",'%04d' %(epoch+1),"cost=",\
                  "{:.9f}".format(avg_cost))
print("Optimization Finished!")
    #Test model
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    #Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print("Accuracy",accuracy.eval({X:mnist.test.images,y:mnist.test.labels}))
#result:
#https://zhuanlan.zhihu.com/p/22410917
```
原帖中的cost数值是错的




```
import tensorflow as tf
W=tf.Variable(tf.zeros([2,1]),name="weights")
b=tf.Variable(0.,name="bias")
def inference(X):
    return tf.matmul(X,W)+b
def loss(X,Y):
    Y_predicted=inference(X)
    return tf.reduce_sum(tf.squared_difference(Y,Y_predicted))
def inputs():
    weights_age=[[84,46],[73,20],[65,52],[70,30]]
    blood_fat_content=[354,190,405,263]
    return tf.to_float(weights_age),tf.to_float(blood_fat_content)
def train(total_loss):
    learning_rate=0.000001;
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
def evaluate(sess,X,Y):
    print(sess.run(inference([[80.,25.]])))
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    X,Y=inputs()
    total_loss=loss(X,Y)
    train_op=train(total_loss)
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    training_iterations=1000
    for iter in range(training_iterations):
        sess.run([train_op])
        if iter%100 == 0:
            print("loss:",sess.run([total_loss]))
    evaluate(sess,X,Y)
    coord.request_stop()
    coord.join(threads)
    sess.close()



----------------------------
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/usr/local/lib/python3.6/site-packages/tensorflow/examples/tutorials/mnist",one_hot=True)
#不用网络下载(因为被墙了下载会失败),而是将下载好的数据放到对应文件夹。
sess=tf.InteractiveSession()
X=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
pred=tf.nn.softmax(tf.matmul(X,W)+b)
target=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(target*tf.log(pred),reduction_indices=[1]))

train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.global_variables_initializer().run()
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({X:batch_xs,target:batch_ys})

correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(target,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({X:mnist.test.images,target:mnist.test.labels}))


-----------------------
import tensorflow as tf
from numpy.random import RandomState
batch_size = 8
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_label = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
w1= tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)
#定义损失函数使得预测少了的损失大，于是模型应该偏向多的方向预测。
#换句话说,对于y>y_label的情况，惩罚较小,所以更倾向于y>y_label
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y, y_label), (y - y_label) * loss_more, (y_label - y) * loss_less))
#如果y大,(y - y_label) * loss_more
#如果y_label大,(y_label - y) * loss_less

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
rdm = RandomState(1)
X = rdm.rand(128,2)
Y = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1, x2) in X]
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % 128
        end = (i*batch_size) % 128 + batch_size
        sess.run(train_step, feed_dict={x: X[start:end], y_label: Y[start:end]})
        if i % 1000 == 0:
            print("After %d training step(s), w1 is: " % (i))
            print(sess.run(w1), "\n")
    print("Final w1 is: \n", sess.run(w1))

-------------------
#保存model
import tensorflow as tf
v1=tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
v2=tf.Variable(tf.constant(1.0,shape=[1]),name="v2")
result=v1+v2
init_op=tf.global_variables_initializer()
saver=tf.train.Saver()
with tf.Session()  as sess:
    sess.run(init_op)
    saver.save(sess,"model.ckpt")

------------
import tensorflow as tf
W=tf.Variable(tf.zeros([2,1]),name="weights")

print("-")
print(W)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    W_val = sess.run(W)
    print("--")
    print(W_val)
    print("---")
    print(W)

"""
-
<tf.Variable 'weights:0' shape=(2, 1) dtype=float32_ref>
--
[[ 0.]
 [ 0.]]
---
<tf.Variable 'weights:0' shape=(2, 1) dtype=float32_ref>
"""

```


```
TensorFlow print variable 的值 是这样的： 
import tensorflow as tf 
W=tf.Variable(tf.zeros([2,1]),name="weights")
print("-") 
print(W) 
with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer()) 
    W_val = sess.run(W) 
    print("--") 
    print(W_val) 
    print("---") 
    print(W)
```


TensorFlow与我们正常的编程思维略有不同：TensorFlow中的语句不会立即执行；而是等到开启会话session的时候，才会执行session.run()中的语句。如果run中涉及到其他的节点，也会执行到。

Tesorflow模型中的所有的节点都是可以视为运算操作op或tensor

http://blog.csdn.net/zhouyelihua/article/details/62210357

回归和保存model


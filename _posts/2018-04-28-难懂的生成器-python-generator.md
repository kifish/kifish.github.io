---
layout: post
tags: [python,advanced]
---


由于函数无法保存状态，故需要一个全局变量配合函数保存状态。

>因为generator可以在执行过程中多次返回，所以它看上去就像一个可以记住执行状态的函数，利用这一点，写一个generator就可以实现需要用面向对象才能实现的功能。


generator有send方法，并且有返回值，如下例的receive。send方法有点类似next()，但前者可以调用参数，并且后者往往是yield var

```python
def simp():
    firstnum = 0
    while True:
        num = firstnum;
        while True:
            num+=10
            firstnum = yield num
            if firstnum:
                break

it=simp()
print(it.next()) # 10
print(it.next()) # 20
print(it.next()) # 30

print(it.send(200)) # 210
print(it.next()) # 220
print(it.next()) # 230
```

```python
def gen():
    idx=0
    while True:
        receive = yield idx
        for _idx,val in enumerate([1,2,3,4,5,6]):
            if _idx < receive:
                continue
            if _idx > receive:
                break
            print('receive_idx :',receive)
            print('item :',val)


g=gen()
g.send(None) # 执行到 receive = yield idx 即返回
g.send(1) # 继续从 receive = yield idx 开始执行，此时receive值为1
g.send(2) # 继续从 receive = yield idx 开始执行，此时receive值为2
g.send(3) # 继续从 receive = yield idx 开始执行，此时receive值为3

```

输出：   
```
receive_idx : 1
item : 2
receive_idx : 2
item : 3
receive_idx : 3
item : 4
```


把值传入或者传出都是用的yield表达式。


注：如果send的值比列表元素的总数大，则会打印最后一个值


上面的的写法似乎多此一举，但是如果在GUI程序中，这样生成器的用法可以用来处理：
点击一下“next”按钮，就处理下一行数据，并返回结果。

```python
def printvar():
    data = yield 3  #line1
    print('in----',data)  #line2
    data = yield   #line3
    print('in----',data)

x = printvar()
print(next(x))

#3

#只执行到line1就返回了

```

```python
def printvar():
    data = yield 3  #line1
    print('in------',data)  #line2
    data = yield   #line3
    print('in----',data)

x = printvar()
print(next(x))
x.send(4)
#3
#in------ 4
```
send改变了程序的流向,x.send(4)从line1开始执行，data被赋值为4,在line3返回(未执行line3)。若有next(x)则会继续从line3开始执行。





```python
def printvar():
    data = yield 3  #line1
    print('in------',data)  #line2
    data = yield   #line3
    print('in----',data) #line4

x = printvar()
print(next(x))
x.send(4)
x.send(5)
```

```
3
in------ 4
in---- 5
Traceback (most recent call last):
  File "shanchu.py", line 13, in <module>
    x.send(5)
StopIteration
```
next(x)从line1开始执行，并返回。
x.send(4)从line1开始执行,执行到line3之前(不执行line3)并返回。   
x.send(5)从line3开始执行,并要寻找到下一个yield表达式以便返回,当执行完line4,之后已经没有yield表达式了，所以x.send(5)会报错。

send 和 next都只消耗一个yield表达式(?),但前者需要有两个yield表达式,第二个yield表达式作为返回点。

send到底消耗几个yield表达式,从后面的结果来看感觉是两个,这里感觉是1个。难道和相近的send或是next有关？这样排列组合有四种情况。有空试一下。

参考：https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001432090171191d05dae6e129940518d1d6cf6eeaaa969000

send,需要遇到两个yield表达式，第一个yield表达式用于传入值，第二个yield表达式用于传出值，但第二个yield表达式不会被消耗掉，下次next或send会从第二个yield表达式的地方接着执行。

>当第一次send（None）（对应c.send(None)）时，启动生成器，从生成器函数的第一行代码开始执行，直到第一次执行完yield（对应n = yield r）后，跳出生成器函数。这个过程中，n一直没有定义。
下面运行到send（1）时，进入生成器函数，注意这里是从n = yield r开始执行，把1赋值给n，但是并不执行yield部分。下面继续从yield的下一语句继续执行，然后重新运行到yield语句，执行后，跳出生成器函数。

var x = yield r
如果调用的是send(input),第一次遇到yield表达式，相当于x = input,第二次遇到yield表达式 yield r (返回r)








```python
def gui_test(classifier):
    run_idx = 0
    pred_or_unpred_res = []
    cnt2 = 0 #无法预测
    fread = codecs.open('./rawdata/labeled_data_for_test.txt', 'r', 'utf-8')
    cnt = 0
    cnt3 = 0
    #clear previous prediction files

    if os.path.exists('./train_data'):
        shutil.rmtree('./train_data')
        os.mkdir('./train_data')
    get_name_pinyin.id2name_pinyin = {}
    get_name_pinyin.id2name_cn = {}
    get_name_pinyin.get_id2name_pinyin()

    while True:
        receive = yield run_idx
        print(receive)
        #location1
        for line in fread.readlines():
            cnt += 1  #location2
             ...
```
把fread放在yield之前，那么fread.readlines在进行第二次迭代的时候就会为空。所以fread应该放在#location1的位置;如果要用到cnt那么应该在yield之后重置cnt。


```
按照鸭子模型理论，生成器就是一种迭代器，可以使用for进行迭代。     
第一次执行next(generator)时，会执行完yield语句后程序进行挂起，所有的参数和状态会进行保存。再一次执行next(generator)时，会从挂起的状态开始往后执行。在遇到程序的结尾或者遇到StopIteration时，循环结束。      
可以通过generator.send(arg)来传入参数，这是协程模型。    
可以通过generator.throw(exception)来传入一个异常。throw语句会消耗掉一个yield。可以通过generator.close()来手动关闭生成器。      
next()等价于send(None)     
```
但是如果先用send，再用next,似乎send不会给next保存参数和状态     

```python
def gen():
    idx=0
    while True:
        receive = yield idx
        for _idx,val in enumerate([1,2,3,4,5,6]):
            if _idx < receive:
                continue
            if _idx > receive:
                break

            val += 10
            print(val)
            print('receive_idx :',receive)
            yield val


g=gen()

g.send(None)
g.send(2)
print("----")
print(next(g))
print("----")


```

```
13
receive_idx : 2
----
0
----
```

实际上我的理解:
>如果先用send，再用next,似乎send不会给next保存参数和状态      

错误


```python
def gen():
    idx=0
    while True:
        receive = yield idx
        for _idx,val in enumerate([1,2,3,4,5,6]):
            if _idx < receive:
                continue
            if _idx > receive:
                break

            val += 10
            print(val)
            print('receive_idx :',receive)
            yield val


g=gen()

g.send(None)
print(g.send(2))
print("----")
print(next(g))
print("----")
```


```
13
receive_idx : 2
13
----
0
----
```
send会执行到下一个yield，并且包括下一个yield,然后暂停（相当于断点）


```python
def gen():
    while True:
        receive = yield #line1
        print('a----receive:',receive)
        var = receive + 1
        print('b----receive:',receive)
        _ = yield var

g = gen()
next(g) #执行到line1

var = g.send(2) #从line1继续执行(包括line1)
print('var :',var)
next(g)

var = g.send(20)
print('var :',var)

```

```
a----receive: 2
b----receive: 2
var : 3
a----receive: 20
b----receive: 20
var : 21

```

next 和 send里的yield表达式其实是一样的。

next:
yield var    
相当于 _ = yield var
      _ 会作为next的返回值
send:
x = yield var  #传入var的值并且赋值给x
实际上去掉var    
x = yield #一样赋值给x

可以看到 next和send实际上都是yield表达式   
只不过next中返回了_的值，但是disgard了_,并没有在上下文里保存。    
而send保存了值，并且没有立即进入中断，而是执行到了下一个yield表达式（包括），执行完之后，进入中断。

画个图来表达：   

               next                                                            send     
                .                                                                |        
                |                                                                |            
                |                                                                .            
         (default = output _ )(也可以指定output var,从而保留该值)  (=) yield (input var)(default = None)  #如果给出了input var,还可以设置input var的默认值



下图见SO  


                                     next() 消耗了一次yield表达式
                                 ==========       yield      ========
                                 Generator |   ------------> | User |
                                 ==========                  ========



                                     send() 消耗了两次yield表达式
                                 ==========       yield       ========
                                 Generator |   ------------>  | User |
                                 ==========    <------------  ========
                                                   send


详细的可以看这篇:     
http://kissg.me/2016/04/09/python-generator-yield/  


-http://www.cnblogs.com/jessonluo/p/4732565.html

send其实是协程
先读这篇：   
-https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001432090171191d05dae6e129940518d1d6cf6eeaaa969000    

>协程的特点在于是一个线程执行，那和多线程比，协程有何优势？
>最大的优势就是协程极高的执行效率。因为子程序切换不是线程切换，而是由程序自身控制，因此，没有线程切换的开销，和多线程比，线程数量越多，协程的性能优势就越明显。

个人理解是线程共享了代码段，切换的时候，需要用到栈；而协程并没有用到栈，共享了变量，只是CPU执行代码的顺序改变了，可能类似汇编里的跳转(?)


参考：
-https://stackoverflow.com/questions/19302530/python-generator-send-function-purpose     
-http://python.jobbole.com/81911/     
-http://devarea.com/python-understanding-generators/#.WuRvjNOFPOQ        
-https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/     

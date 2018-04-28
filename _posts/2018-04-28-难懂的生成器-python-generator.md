---
layout: post
tags: [python,advanced]
---
见SO   
```
==========       yield      ========
Generator |   ------------> | User |
==========                  ========
```

```
==========       yield       ========
Generator |   ------------>  | User |
==========    <------------  ========
                  send
```

generator有send方法，并且有返回值，如下例的receive。send方法有点类似next（），但前者可以调用参数，并且后者往往是yield var

```Python
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

```Python
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
g.send(None)
g.send(1)
g.send(2)
g.send(3)

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

注：如果send的值比列表元素的总数大，则会打印最后一个值


上面的的写法似乎多此一举，但是如果在GUI程序中，这样生成器的用法可以用来处理：
点击一下“next”按钮，就处理下一行数据，并返回结果。

```
def printvar():
    data = yield 3  #line1
    print('in----',data)  #line2
    data = yield   #line3
    print('in----',data)

x = printvar()
print(next(x))

#3

```

```
def printvar():
    data = yield 3  #line1
    print('in----',data)  #line2
    data = yield   #line3
    print('in----',data)

x = printvar()
print(next(x))
x.send(4)
#3
#in---- 4
```
实际上line2并没有被执行，因为send改变了程序的流向，从line3开始执行了。



```
def printvar():
    data = yield 3  #line1
    print('in----',data)  #line2
    data = yield   #line3
    print('in----',data)

x = printvar()
print(next(x))
x.send(4)
x.send(5)
```

```
3
in---- 4
in---- 5
Traceback (most recent call last):
  File "shanchu.py", line 13, in <module>
    x.send(5)
StopIteration
```
实际上x.send(4)之后已经没有yield了，所以x.send(5)会报错。



```
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

```
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


-http://www.cnblogs.com/jessonluo/p/4732565.html

send其实是协程
先读这篇：   
-https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001432090171191d05dae6e129940518d1d6cf6eeaaa969000    

>协程的特点在于是一个线程执行，那和多线程比，协程有何优势？
>最大的优势就是协程极高的执行效率。因为子程序切换不是线程切换，而是由程序自身控制，因此，没有线程切换的开销，和多线程比，线程数量越多，协程的性能优势就越明显。

个人理解是线程共享了代码段和数据段，因此切换的时候，需要用到栈；而协程并没有用到栈，共享了变量，只是CPU执行代码的顺序改变了，可能类似汇编里的跳转(?)


参考：
-https://stackoverflow.com/questions/19302530/python-generator-send-function-purpose
-http://python.jobbole.com/81911/
-http://devarea.com/python-understanding-generators/#.WuRvjNOFPOQ

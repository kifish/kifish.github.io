---
layout: post
tags: [python]
published : false
---





http://www.yangyingming.com/article/19/   

>python中list变量作为全局变量时，在函数中可以直接修改。
而普通变量则需要先在函数中global声明，否则会报错。
例如：
a = 1
def fun():
global a
a = 2
而list：
b = [1,2]
def fun():
b[0] = 2
在函数中直接修改list则是可以的。
原因是：
普通变量如果在函数中赋值
a = 2
会有歧义。因为它既可以是表示引用全局变量a，也可以是创建一个新的局部变量，所以在python中，默认它的行为是创建局部变量，除非显式声明global。
而对列表list变量进行赋值
b[0] = 2
则不会有歧义。它是“明确的”，因为如果把b当作是局部变量的话，它会报KeyError，所以它只能是引用全局的b,故不需要多此一举显式声明global。





python的陷阱:
通过缩进判断for循环作用域的, for x in l:

但是x在for循环结束之后依然有效。

python的else 匹配是通过缩进 而不是匹配给上一个if。

但是 else 的匹配 和缩进关系不大 ,主要是匹配给上一个if.

https://colab.research.google.com/drive/15ipbb4fJ-ItEMX_rSn8kEkArtx0sV4dg#scrollTo=IIjZNTbfqbom






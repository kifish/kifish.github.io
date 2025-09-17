
python版代码可参考：
http://blog.csdn.net/dream_angel_z/article/details/46355803    
伪代码：http://f.dataguru.cn/thread-258437-1-1.html   
参考：http://blog.csdn.net/lfdanding/article/details/50755919  
 上述链接中的代码总体来说挺清晰的，有2个地方比较难理解。  

% Apriori rule: if any subset of items are not in frequent itemset do not
% consider the superset (e.g., if {A, B} does not have minSup do not consider {A,B,\*})  
```
if sum(ismember(nchoosek(Combinations(j,:),steps-1),TOld,'rows')) - steps+1>0      
```
  ismember的用法：http://blog.csdn.net/yes1989yes/article/details/73302766
这一行应该是判断字串是否包含，如果包含再进行频繁集连接。
```
% Calculate the support for the new itemset(e.g.{A,B})       
S = mean((sum(transactions(:,Combinations(j,:)),2)-steps)>=0);
```
接下来就是求support，实际调试的时候S=0，感觉可能存在问题，使用上述链接中的代码和数据进行单步运行，发现输出结果和上述链接中的不符。
可能问题就出在这。
我动手实现了这一算法，详见github。
学到了细节处理，可用char()实现cell转str



![](http://lh3.googleusercontent.com/CEnfGg2FOsvvQ6zfbHM9GIa5aneO-rUpoT3Hc8cIiMOaH5vpwZQfF7YlrwEhqWGs0bzVwkbl6UtMYW3dykTY46pJ5vrEaoI54WfZv19fZ0CJ8gbfgrY5P9gIcX7AhHwk8Nhl6cjm  "optional title")

matlab也可以像C语言一样格式化输出 sprintf
 http://blog.csdn.net/u013457167/article/details/48805217
一般还要注意修改0-index/1-index

The number of rules：
At first, we can divide the m items into two sets. One set is for conditions, and the other is for conclusions. If the number of items in the condition set is i .Then, the number of items in the conclusion set is m-i . And the value of i can be 1,2,3,……,n-1. Therefore, the total number of the rules can be obtained by multiplying the two numbers of the separate combinations and finally summing the products.  
In conclusion, the number of rules can be expressed as follows: 

![](http://lh6.googleusercontent.com/3HgxqqM-kccnNSECA7Qj5tkcQHax6uXylA7kBEk4pHUouxG_2IAnmBtTCakEp8CTEuJO5YugBeNUtwNVCFaAfwro1u-0bF1X66Lwx_lODfVkXdvAdk_SePs_B_m6853DBJkq62Os)     


![](http://lh6.googleusercontent.com/lA5IQlYhZKXWl7tB2i3frBmhoOZIBvlScVwZ-9DrysOdjjIQqLWZVGSkJUl8KkOGBdePw7KoyHFxNhenW9zp0YFFQWQ2nWIiN3_r3-DI_nbAXBbyT8tGNNqPU3WoVS6_pQGfQZUF)

![](https://lh3.googleusercontent.com/sBfXI98hc5Ru569SMAWccsZiIiT5FQBbnJ1MUHQ_W-YpbDHyA-5tU1aUzMUUmS7eVnvK2XCZcK7Nlo2GEbjD0ZvN_jcEOwdM8pclCXyCrQCHWxuf2sIenoNstMyFEKaqjFn-cR5V)


![](https://lh3.googleusercontent.com/ecIXpPLS_GUCLDn3sZ0a2JIIkSe1wRSBrDbpqbDGSBC5h9gX8IHtaAsJjypofljmsIJA6hyzRKiB1p1xUkHcxILbTVYTtgm5YVOA3JfnP3tba_TEXI4A-LbYbB7v4PYudHgstTdM)

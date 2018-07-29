可以看下这个博客
http://blog.csdn.net/hlx371240/article/details/41730579
k-fold cross validation实现：
http://blog.csdn.net/enjolras_fuu/article/details/72190139
 
 
matlab实现见github。
matlab中函数返回的向量，原来是行向量，返回也是行向量，原来是列向量，返回也是列向量。
出现一些问题：做cross validation的时候，error rate 是随着lambda在变的，但是对testdata做prediction的时候 error rate却不会因为lambda变化。发现是梯度迭代写错了,1.逻辑判断条件不对，2. g<1e-3 应该写成norm（g）<1e-3 。最好改成max(abs(g))<1e-3，这样收敛会快一点。



https://docs.google.com/document/d/1iJIJKjq7DPNs6PNCEzNWF7xJdTqT_R7A6fLlK1uQAiQ/edit

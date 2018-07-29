---
layout: post
tags: [matlab,ml]
published : true
---



As we have seen in class this quarter, the nearest neighbor algorithm is a very simple, yet very competitive classification algorithm. It does have one major drawback however; it is very sensitive to irrelevant features. 
参考：http://blog.csdn.net/llp1992/article/details/45040685
http://blog.csdn.net/rk2900/article/details/9080821  
http://www.jianshu.com/p/48d391dab189  python实现
https://www.cnblogs.com/tiandsp/archive/2013/03/06/2946310.html  测试数据
 knn.m
function predict = knn(test_data,train_data,train_labels,k)
[m,~] = size(train_data);
diff_mat = repmat(test_data,[m,1])-train_data;
dist_mat = sqrt(sum(diff_mat.^2,2));
[~,idx] = sort(dist_mat,'ascend');
len = min(k,m);
predict = mode(train_labels(idx(1:len)));
test.m
data = load('test.txt');
data_mat = data(:,1:3);
labels = data(:,4);
m = size(data_mat,1);
k = 5;
error = 0;
% 测试数据比例
Ratio = 0.1;
numTest = Ratio * m;
% 归一化处理
maxV = max(data_mat);
minV = min(data_mat);
range = maxV-minV;
newdata_mat = (data_mat-repmat(minV,[m,1]))./(repmat(range,[m,1]));
% 测试
for i = 1:numTest
    predict = knn(newdata_mat(i,:),newdata_mat(numTest+1:m,:),labels(numTest+1:m,:),k);
    fprintf('测试结果为：%d  真实结果为：%d\n',[predict labels(i)])
    if(predict~=labels(i))
        error = error+1;
    end
end
fprintf('准确率为：%f\n',1-error/(numTest))
测试数据：



[1]
[2]










 level有点难理解,我把level理解为可能的所选feature的最大个数.(forward selection).例如,第四次大循环的可能的所选feature的最大个数即为4.
对于feature的搜索可以考虑:
增支;
剪支;
先增支再剪支;
 
选择real feature可以copy多次data,然后随机删除5%.
加速,可以不计算accuracy,发现错的样本过多,直接退出.

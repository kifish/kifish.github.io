---
layout: post
tags: [matlab,ml]
published : true
---


参考:http://blog.csdn.net/jiandanjinxin/article/details/50598155

下面这个比较乱:http://blog.csdn.net/watkinsong/article/details/8234766

http://blog.csdn.net/lijihw_1022/article/details/46622667  这个链接求写协方差矩阵是X'X/(n-1).似乎两种做法:/n或者/n-1都有看到过,可能类似总体std和样本std的关系吧.我觉得应该使用/(n-1)

https://www.cnblogs.com/sweetyu/p/5085798.html

http://blog.jobbole.com/86905/  散点图比较

http://blog.sina.com.cn/s/blog_61b8694b0101jg4f.html  测试数据

MATLAB直接用样本实现主成分分析用有多种方式，但是mathwork公司推荐(1)式，因为princomp在使用时调用的是pca，两者的计算结果一样，而且pca多一项explain，更强大。
[coeff,score,latent,tsquared,explained]= pca(X)      (1)
[COEFF,SCORE,latent,tsquare] = princomp(X)        (2)


[coef,score,latent,t2] = princomp(x);（个人观点）：

princomp/pca 会自动对数据进行减去均值的归一化.

x：为要输入的n维原始数据。带入这个matlab自带函数，将会生成新的n维加工后的数据（即score）。此数据与之前的n维原始数据一一对应。

score：生成的n维加工后的数据存在score里。它是对原始数据进行的分析，进而在新的坐标系下获得的数据。他将这n维数据按贡献率由大到小排列。（即在改变坐标系的情况下，又对n维数据排序）

latent：(协方差特征值而已,大小对应了方差,也就是所贡献的信息量大小)是一维列向量，每一个数据是对应score里相应维的贡献率，因为数据有n维所以列向量有n个数据。由大到小排列（因为score也是按贡献率由大到小排列）。

coef：是系数矩阵。通过cofe可以知道x是怎样转换成score的。

则模型为从原始数据出发：
score= bsxfun(@minus,x,mean(x,1)) * coef;
(作用：可以把测试数据通过此方法转变为新的坐标系)
逆变换：
x= bsxfun(@plus,score*inv(coef),mean(x,1))


score实际上就是原始数据经过pca变换后(降维)的数据,可以直接拿来用.
实际上pca是可逆的,也就是说只要记录了coef(如果初始做归一化了还需要mean),即可以逆变换.
这个逆变换之后的数据,相当于对于原始数据做了滤波.



matlab里面的pca默认会做减去均值的预处理，我动手实现了pca，可以选择不做减去均值的预处理。

变量名命名有点混乱。有空再改吧。

main.m



```matlab
 %input
rawdata = transpose(input);
mean_data = mean(rawdata);
th = 95;%%这里th可以改
flag= 0;
%%flag=1,代表预处理减去mean
%flag=0 则不减去mean
FiltedData = PCA_Filter_Noise(rawdata,th,flag);
if flag==1
    FiltedData = ones(size(FiltedData,1),1)*mean_data+FiltedData;
end
FiltedData = FiltedData';

```

PCA_Filter_Noise.m  

```matlab
%% 尝试用PCA去除发射电流噪声
% PCA 原始数据建立
% RawData:m*n维，样本数m,样本维度为n
%   行向量为一个样本包含的各分量，1*n
%   列向量为样本中某个变量的多次观测值
function [ FiltedData] = PCA_Filter_Noise(RawData,th,flag)
     [TransData,Principal_Componentvec,mean] = PCA_Reduce_Dimension(RawData,th,flag);
     ReconstructedData = TransData*Principal_Componentvec';
     FiltedData = ReconstructedData;          
end
```

PCA_Reduce_Dimension.m    
```
% PCA  降低原始数据的维度 原始数据建立
% Normalized_Data:m*n维，样本数m,样本维度为n
%   行向量为一个样本包含的各分量，1*n，各分量的量刚应该一致
%   列向量为样本中某个变量的多次观测值

function [ TransData,Principal_Componentvec,mean] = PCA_Reduce_Dimension(data,threshold,flag)
    if (0<=threshold)&&(threshold<=100)
        threshold = threshold/100;
    else
        error('阈值输入错误！');
    end
    if flag ==1
        [EigenVector,Score,EigenValue,Tsquared,RatioOfEigenValue,mean] = pca(data);
        sum_RatioOfEigenValue = cumsum(RatioOfEigenValue./100);
    else
        [ EigenVector,Score,EigenValue,sum_RatioOfEigenValue,mean ] = pca2(data);
    end
    id_EigenValue = find(sum_RatioOfEigenValue >= threshold);

    NumComponents = id_EigenValue(1);%从最大的特征值开始选取,id_EigenValue(1)是所需选取的最后一个特征值的编号
    Principal_EigenValue = EigenValue(1:NumComponents);
    Principal_Componentvec = EigenVector(:,1:NumComponents);% 列向量为特征向量
%     % 输出所有特征值和特征向量
%     fprintf('特征值: %f \n')
%     disp(EigenValue);
%     fprintf('各特征值的贡献率: %f \n')
%     disp(RatioOfEigenValue);
%     fprintf('特征向量: %f \n')
%     disp(EigenVector);
%     % 输出满足条件的主分量和对应的特征值和特征向量
%     fprintf('方差累计贡献率: %f \n');
%     disp(sum_RatioOfEigenValue);
%     fprintf('方差累计贡献率大于 %f的特征值为: %f \n');
%     disp(threshold);disp(Principal_EigenValue);
%      fprintf('方差累计贡献率大于 %f的特征向量为: %f \n');
%      disp(threshold);disp(Principal_Componentvec);

    %RestructedData = Normalized_Data*Principal_Component;只有这里的
    %Normalized_Data做了减去均值的归一化(应该做),上下两行的结果才一致
    TransData = Score(:,1:NumComponents);
end

```


pca2.m    

```
function [ eigVect,Score,lambda,sum_ratio,meanValue ] = pca2(data)
%
meanValue = mean(data);
%方法1 C = cov(data)
%方法2
%normData = data- repmat(meanValue,[size(data,1),1]);    %
%中心化样本矩阵,测试如果不做中心化,有什么影响,那么score肯定和做了中心化求出score有不同
%实际上pca默认做了中心化
normData = data;
CovMat = (normData'*normData)/(size(normData,1)-1);
[eigVect,eigVal] = eig(CovMat);%求取特征值和特征向量
lambda = eigVal(eigVal~=0);
[lambda,idx] = sort(lambda,'descend');
eigVect = eigVect(:,idx);
sum_lambda = sum(lambda);
lambda_ratio = lambda/sum_lambda;
sum_ratio = cumsum(lambda_ratio);
Score = normData *eigVect;
end
```



test.m
```
testdata = [7 26 6 60;
1 29 15 52;
11 56 8 20;
11 31 8 47;
7 52 6 33;
11 55 9 22;
3 71 17 6;
1 31 22 44;
2 54 18 22;
21 47 4 26;
1 40 23 34;
11 66 9 12;
10 68 8 12];
[EigenVector,score,latent,tsquare] = pca(testdata ); %调用pca分析函数
[ eigVect,Score,lambda,sum_ratio,meanValue ] = pca2(testdata);

```

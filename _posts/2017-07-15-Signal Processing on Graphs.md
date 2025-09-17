引言：感觉这三篇文献的阅读顺序应该是SPM-GSP-color----àDepth Map Denoising using Graph-based Transform and Group Sparsity----àDavid Shuman_The Emerging Field of Signal Processing on Graphs.
难度也是依次递增，最后那篇综述看的很慢.
图像处理的侧重点感觉和计算机视觉有一点不一样（当然也可以认为后者包含前者）。
图像处理偏重：去噪、去雾、分割、分类、人脸识别。（以下图片参考某博客，一图胜前言）



[1]
[2]
（超分辨率重建）




（Inpainting顾名思义可能会弄错，Inpainting is the process of reconstructing lost or deteriorated parts of images and videos. In the museum world, in the case of a valuable painting, this task would be carried out by a skilled art conservator or art restorer. In the digital world, inpainting (also known as image interpolation or video interpolation) refers to the application of sophisticated algorithms to replace lost or corrupted parts of the image data (mainly small regions or to remove small defects).
计算机视觉偏重：物体检测、定位、识别、场景分类。（边缘检测好像图像处理和计算机视觉都有。）




 
（摘自cs231n—lecture--9）
文献里说的是KNN cluster. 但是机器学习里一般把KNN称作分类算法、把K-means称作聚类（因为前者有label，后者无label）参见：
http://www.tuicool.com/m/articles/qamYZv
当然文献里的用法感觉更像是聚类。
Depth Map Denoising using Graph-based Transform and Group Sparsity
几个概念：
Depth map




 
相机会记录距离：离视角近的更黑一些。（当然也有其他做法。）
Nonlocal和global
我原来以为nonlocal顾名思义就是全局了，实际上不是，意思是非局部。有点类似这样：
def make_counter():
    count = 0
    def counter():
        nonlocal count
        count += 1
        return count
    return counter
def make_counter_test():
  mc = make_counter()
  print(mc())
  print(mc())
  print(mc())
 
 
定义：
In natural images, usually we can find many similar patches to a given path, which can be spatially far from it. This is called non local self-similarity.
 
文献的脉络
 
背景及意义：depth sensors成本降低，越来越普及。这类设备可以获得三维几何信息并选定视角投影为二维图像。在设备获得信息的时候可能会混入噪声，因此我们可以做一些应用比如去噪。
相关工作：非局部的图像去噪利用了类似的模式会在图像中重复的假设。基于这种依赖关系，我们可以聚类patches，然后恢复图片。
字典学习，我认为其实有点类似找一个空间的特征向量的。找到足够的特征向量，特定向量即可用特征向量线性表示。
 
文献中主要用到了以下两个性质：
1. 分段光滑   piecewise smooth characteristic
2. 非局部自相似性  nonlocal self-similarity
主要是以下三步：
1. 先聚类patches然后平均（利用nonlocal similar geometry，平均就是字典学习的过程）
2. 图变换（GBT），找出基向量（即字典学习）
3. 迭代做group sparsity
 
注意：step1，2是同时做的。个人理解是平均后的patch就是基向量了。举例来说，比如图中有一个足球和绿色草地的背景，可能基向量就是由足球和草地的patches组成了。下文的内容没有考虑color。
 
思考：这里的聚类是用KNN，用的是普通的几何距离（因为我们没有考虑颜色），如果考虑颜色（RGB）和距离，要不要用高斯核呢？
做GBT的时候好像和（PDE-Based Graph Signal Processing for 3-D Color Point Clouds
）有点不太一样。前者是



[]
后者是

[]





把patch展成向量，对基做投影。加权GBT就是把相似度代入，这也是本文的创新点。
接下来就是论文最核心的部分了，简洁而优雅：通过L0范数倾向稀疏性来做稀疏编码，从而达到去噪效果。写成矩阵形式就是group sparsity------- reconstruct the clean manifold geometry。
注意：L0范数非凸，属于NP-hard问题。可以换成L1，倾向稀疏性的性质不变，变出凸优化问题。然而应该不能退化成L2，因为L2倾向于系数平滑分布，尽管L2很好计算和优化。L1范数优化求解计算量还是很大，文献里提出transform spectrum shrinkag，我的猜测是若系数小于一个阈值，就设为0。某种程度上也可以理解为让低频图信号通过，滤除高频信号即噪声。
图像恢复的时候，patch可能重叠，就加权处理，系数大的权重大。最后一步即迭代，有点类似小明做完试卷再重做检查一遍然后再重做检查一遍如此迭代下去…(这个迭代是外循环)
做实验，主客观指标验证文献的算法很棒。
D. Subjective quality这一部分的第二段有一个笔误：One the other hand 这里应该是On。
因为算法用到了以下两个性质：
1分段光滑   piecewise smooth characteristic
2非局部自相似性  nonlocal self-similarity
因此击败了其他算法。
未来的工作就是要用直接来自传感器的depth map来验证，而不是加高斯白噪声。
PDE-Based Graph Signal Processing for 3-D Color Point Clouds
主要内容：
通过3D点云来做物体的identification and examination(因为3D可以比2D保存更细致的信息).可以用来做cultural heritage preservation.
传统的信号处理方法不能适用于3D点云，因为缺乏结构化信息。
两部分内容：
1.3D点云下的PDE--->PdE
2针对艺术品的应用
 
获得3D点云主要有两种方法:
1. 摄影测量法
2. 激光扫描(更贵也更精确)
 
 
点云缺乏拓扑结构,因此传统的处理2D图像的方法难以运用。
文献想通过点云来构造图。KNN用来聚类（不是分类）。边加权通过高斯核即可。用一个小球包住邻近点并投影到切平面。一个cell里取点的投影的平均。
Patch是一个很重要的概念，正方形，n个cell组成一个patch。
我理解的PdE：考虑了相似度（某个空间的距离，这里是用高斯核，可能是因为高斯核对应了无穷维空间，表达能力很强）对微分（实际上这时候已经没有梯度的概念了，只有微分，PdE英文里很准确difference）的影响。由于是很多点，所以是多维微分，展成向量。
这篇文献里面最吸引我的地方是下面这部分内容：
其实《机器学习基石》讲的也是这个意思：我们的数据集其实有噪声的，假设真实分布是f，那么我们通过minimize error在假设空间里寻找g
来逼近f，从而去除噪声。 文献通过拉格朗日极值法求解，可以通过迭代，但不是梯度下降了。
下面就是讲应用了：用于历史文物或者艺术品的3D数字化、保存、上色、修补等。用的方法是，让邻近点的颜色和已知点的颜色的某种距离为0（不知道我这里有没有理解错？）
文献里的滤波是为了减少点的数量，计算更快。
值得注意的是，inpainting和colorization公式里只是p的值不一样。图像分隔原理是若点feature的距离为某一定值内则皆为同一块。
 
 
The Emerging Field of Signal Processing on Graphs
这一篇综述我看的很慢，还在看。
图信号不同于传统信号,传统信号只有时间先后,图信号则有拓扑结构.
因此传统的信号处理方法未必适用于图信号处理.。
文献想把在信号处理中的知识迁移到图信号。联想以前学习的到知识：模拟信号（无第二类间断点）一般可以作傅里叶或者拉普拉斯变换，在频域里分解后分析，频率对应了变化的快慢。图信号也可以做类似处理，找特征值（即频率，对应了相邻点feature变换的快慢）。未完待续…
 
 
References：
Hu, Wei, et al. "Depth map denoising using graph-based transform and group sparsity." IEEE, International Workshop on Multimedia Signal Processing IEEE, 2013:001-006.
https://baike.baidu.com/item/inpainting/1305883?fr=aladdin
https://en.wikipedia.org/wiki/Inpainting
https://en.wikipedia.org/wiki/Depth_map
http://blog.csdn.net/tiandijun/article/details/41578175（这个博客可以好好看看）
Narang, et al. "The Emerging Field of Signal Processing on Graphs." IEEE Signal Processing Magazine 30.3(2013):83-98.
Lozes, Francois, A. Elmoataz, and O. Lezoray. "PDE-Based Graph Signal Processing for 3-D Color Point Clouds : Opportunities for cultural heritage." IEEE Signal Processing Magazine 32.4(2016):103-111.

# 1. 计算机视觉调研报告


## 1.1. 定义
1. 从数字图像中提取信息的科学领域
2. 构建能理解图像内容的算法，并将其用于其他应用。


## 1.2. 图像处理基础
### 1.2.1. 颜色
1. 颜色空间：RGB（属于线性颜色空间）、CIE-XYZ(属于线性颜色空间）、HSV（属于非线性颜色空间）。
2. 
3. 白平衡：是将传感器接收到的图片数据调整至合适的呈现中性的颜色（灰、白等等）的过程。
   相关算法：Von kries Method（Gray Card Method）、Gray World Assumption、Brightest Pixel Assumption.

### 1.2.2. 图像采样和量化
1. 图像类型：
+  黑白图像：像素是黑色（0）或者白色（1）
+   灰阶图像：像素范围在0（黑色）~255（白色）
+   彩色图像：有多个颜色通道，每张图片可以在不同的颜色模型 (RGB, LAB, HSV) 上呈现，每个颜色通道值的范围取决于所选的颜色模型。
2. 直方图（Image Histograms）
  
   直方图用于测试灰度图的强一个特定的像素值 (0-255) 在图像中出现了几次。





### 1.2.3. 卷积和相关

+ 卷积（Convolution）：

  可看作是加权求和的过程，使用到的图像区域中的每个像素分别于卷积核(权矩阵)的每个元素对应相乘，所有乘积之和作为区域中心像素的新值。


  <div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=s[k,l]&space;=&space;\sum_{m=-\infty}^{\infty}\sum_{n=-\infty}^{\infty}f[m,n]g[k-m,l-n]" target="_blank"><img src="https://latex.codecogs.com/png.latex?s[k,l]&space;=&space;\sum_{m=-\infty}^{\infty}\sum_{n=-\infty}^{\infty}f[m,n]g[k-m,l-n]" title="s[k,l] = \sum_{m=-\infty}^{\infty}\sum_{n=-\infty}^{\infty}f[m,n]g[k-m,l-n]" /></a></div>
+ 相关（Correlation）：

  等于卷积计算时内核没有翻转的结果。

  <div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=r[k,l]&space;=&space;\sum_{m=-\infty}^{\infty}\sum_{n=-\infty}^{\infty}f[m&plus;k,n&plus;l]g[m,n]&space;=f[n,m]*g^*[-n,-m]" target="_blank"><img src="https://latex.codecogs.com/png.latex?r[k,l]&space;=&space;\sum_{m=-\infty}^{\infty}\sum_{n=-\infty}^{\infty}f[m&plus;k,n&plus;l]g[m,n]&space;=f[n,m]*g^*[-n,-m]" title="r[k,l] = \sum_{m=-\infty}^{\infty}\sum_{n=-\infty}^{\infty}f[m+k,n+l]g[m,n] =f[n,m]*g^*[-n,-m]" /></a></div>

 * 区别：

    * 卷积是一个积分，它表示当一个函数在另一个函数上移动的时候的重叠部分。也就是说，卷积是一个过滤操作。

    * 相关比较了两个数据集的相似性。相关性计算了两个输入函 s 数相互移动时的相似性测量值。两个函数匹配都越高，它的结果值越大。也就是说，相关性是两个信号关联性的测量值。

* 共性：
  * 都是线性的，即用每个像素的邻域的线性组合来代替这个像素。
  * 都具有平移不变性（shift-invariant)，即在图像的每个位置都执行相同的操作。

### 1.2.4. 滤波

#### 1.2.4.1. 频率(frequency)

频率（frequency）是波动快慢的指标。图像就是色彩的波动：波动大，就是色彩急剧变化；波动小，就是色彩平滑过渡。

大多数图片既有高频成分又有低频成分，色彩剧烈变化的地方，就是图像的高频区域；色彩稳定平滑的地方，就是低频区域。

#### 1.2.4.2. 滤波器(Filter)

定义：由特定公式导出的卷积核。

规则要求：

1）滤波器的大小应该是奇数，这样它才有一个中心，例如3x3，5x5或者7x7。有中心了，也有了半径的称呼，例如5x5大小的核的半径就是2。

2）滤波器矩阵所有的元素之和应该要等于1，这是为了保证滤波前后图像的亮度保持不变。当然了，这不是硬性要求了。

3）如果滤波器矩阵所有元素之和大于1，那么滤波后的图像就会比原图像更亮，反之，如果小于1，那么得到的图像就会变暗。如果和为0，图像不会变黑，但也会非常暗。

4）对于滤波后的结构，可能会出现负数或者大于255的数值。对这种情况，我们将他们直接截断到0和255之间即可。对于负数，也可以取绝对值。

#### 1.2.4.3. 高通滤波器（High Pass Filter）

作用：边缘提取与增强。边缘区域的灰度变换加大，也就是频率较高。所以，对于高通滤波，边缘部分将被保留，非边缘部分将被过滤。常见的高通滤波器有：Sobel，Laplacian等。

* Sobel边缘检测算子

  Sobel 算子是一个离散微分算子 (discrete differentiation operator)。 它用来计算图像灰度函数的近似梯度。

  用

<div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=f&space;'(x)&space;=&space;f(x&space;&plus;&space;1)&space;-&space;f(x&space;-&space;1)" target="_blank"><img src="https://latex.codecogs.com/png.latex?f&space;'(x)&space;=&space;f(x&space;&plus;&space;1)&space;-&space;f(x&space;-&space;1)" title="f '(x) = f(x + 1) - f(x - 1)" /></a></div>
  

  近似计算一阶差分。

  排序后

<div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=[-1&space;*&space;f(x-1),0&space;*&space;f(x),1&space;*&space;f(x&plus;1)]" target="_blank"><img src="https://latex.codecogs.com/png.latex?[-1&space;*&space;f(x-1),0&space;*&space;f(x),1&space;*&space;f(x&plus;1)]" title="[-1 * f(x-1),0 * f(x),1 * f(x+1)]" /></a></div>

  可提出系数[-1,0,1]。
  
  二维情况下就是：


  <div align=center><img src = 'images/sobel1.png'  width="40%"></img> </div>


  
  <div align=center>  Prewitt 边缘检测算子</div>
  
  中心点 f(x, y) 是重点考虑的，它的权重应该多一些，所以改进成下面这样的：
  
   <div align=center><img src = 'images/sobel2.png'  width="60%"></img></div>
  
  <div align=center>  Sobel 边缘检测算子</div>

  在图像的每一点，结合以上两个结果求出近似 梯度:
  <div align=center> <a href="https://www.codecogs.com/eqnedit.php?latex=G&space;=&space;\sqrt{G_x^2&plus;G_y^2}" target="_blank"><img src="https://latex.codecogs.com/png.latex?G&space;=&space;\sqrt{G_x^2&plus;G_y^2}" title="G = \sqrt{G_x^2+G_y^2}" /></a></div>
  
  
  有时也用下面更简单公式代替:
  
 <div align=center> <a href="https://www.codecogs.com/eqnedit.php?latex=G&space;=&space;\left|G_x&plus;G_y\right|" target="_blank"><img src="https://latex.codecogs.com/png.latex?G&space;=&space;\left|G_x&plus;G_y\right|" title="G = \left|G_x+G_y\right|" /></a></div>

 * Laplacian边缘检测算子
   
   二阶微分算子


 <div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=\triangledown^2f(x,y)=\frac{\partial^2f}{\partial&space;x^2}&plus;\frac{\partial^2f}{\partial&space;y^2}=f(x&plus;1,y)&plus;f(x-1,y)&plus;f(x,y&plus;1)&plus;f(x,y-1)-4f(x,y)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\triangledown^2f(x,y)=\frac{\partial^2f}{\partial&space;x^2}&plus;\frac{\partial^2f}{\partial&space;y^2}=f(x&plus;1,y)&plus;f(x-1,y)&plus;f(x,y&plus;1)&plus;f(x,y-1)-4f(x,y)" title="\triangledown^2f(x,y)=\frac{\partial^2f}{\partial x^2}+\frac{\partial^2f}{\partial y^2}=f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)-4f(x,y)" /></a></div>

   得到四邻域的卷积核：
   
   <div align=center><img src = 'images/laplace1.png'  width="20%"></img></div>
   
   同理可得八邻域的卷积核：
   
   <div align=center><img src = 'images/laplace2.png'  width="20%"></img></div>

#### 1.2.4.4. 低通滤波器（Low Pass Filter）

作用：去噪声和模糊化，边缘平滑，边缘区域将被平滑过渡。
* 均值滤波器(Box Filter)

  这个滤波是一个平滑图像的滤波器，它用一个点邻域内像素的平均灰度值来代替该点的灰度
* 高斯（模糊）滤波器
  
  要使图像变得模糊，可以将其与一个二维高斯滤波器做卷积：

 <div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=f(m,n)=\frac{1}{2\pi\sigma^2}exp(-\frac{m^2&plus;n^2}{2\sigma^2})&space;$$" target="_blank"><img src="https://latex.codecogs.com/png.latex?f(m,n)=\frac{1}{2\pi\sigma^2}exp(-\frac{m^2&plus;n^2}{2\sigma^2})&space;$$" title="f(m,n)=\frac{1}{2\pi\sigma^2}exp(-\frac{m^2+n^2}{2\sigma^2}) $$" /></a></div>


  离散的高斯卷积核H： (2k+1)×(2k+1)维，其元素计算方法为：

 <div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=H_{i,j}&space;=&space;\frac{1}{2\pi\sigma^2}e^{-\frac{(i-k-1)^2&plus;(j-k-1)^2)}{2\sigma^2}}" target="_blank"><img src="https://latex.codecogs.com/png.latex?H_{i,j}&space;=&space;\frac{1}{2\pi\sigma^2}e^{-\frac{(i-k-1)^2&plus;(j-k-1)^2)}{2\sigma^2}}" title="H_{i,j} = \frac{1}{2\pi\sigma^2}e^{-\frac{(i-k-1)^2+(j-k-1)^2)}{2\sigma^2}}" /></a></div>
再将权系数归一化，即保证：

 <div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{i,j}^{2k&plus;1}H_{i,j}=1" target="_blank"><img src="https://latex.codecogs.com/png.latex?\sum_{i,j}^{2k&plus;1}H_{i,j}=1" title="\sum_{i,j}^{2k+1}H_{i,j}=1" /></a></div>

  方差越大，滤波后的图像越平滑。



#### 1.2.4.5. 图像锐化滤波器
 图像的锐化和边缘检测很像，首先找到边缘，然后把边缘加到原来的图像上面，这样就强化了图像的边缘，使图像看起来更加锐利了。这两者操作统一起来就是锐化滤波器了，也就是在边缘检测滤波器的基础上，再在中心的位置加1，这样滤波后的图像就会和原始的图像具有同样的亮度了，但是会更加锐利。

<div align=center> <img src = 'images/SharpnessFilter1.png'  width="60%"> </img> </div>

我们把核加大，就可以得到更加精细的锐化效果:


<div align=center> <img src = 'images/SharpnessFilter2.png'  width="60%"> </img> </div>

另外，下面的滤波器会更强调边缘：
      
<div align=center> <img src = 'images/SharpnessFilter3.png'  width="60%"> </img> </div>

主要是强调图像的细节。最简单的3x3的锐化滤波器如下：

<div align=center> <img src = 'images/SharpnessFilter4.png'  width="60%"> </img> </div>

 实际上是计算当前点和周围点的差别，然后将这个差别加到原来的位置上。另外，中间点的权值要比所有的权值和大于1，意味着这个像素要保持原来的值。

### 1.2.5. 傅里叶变换（Fourier Transform）

从物理效果看，傅里叶变换是将图像从空间域转换到频率域，其逆变换是将图像从频率域转换到空间域。换句话说，傅里叶变换的物理意义是将图像的灰度分布函数变换为图像的频率分布函数。 

## 1.3. 基本问题 

### 1.3.1. 边缘检测（Edge Detection）

边缘检测的目的是检测图像中的不连续部分。直观来讲，图像的大部分语义学和形状信息可以在图像边缘被编码。边缘可以帮助我们提取信息、识别物体、恢复几何和视角。

图片上的边缘主要有四个可能的来源：不连续的表面（表面角度突然改变）、深度不连续（一个表面重叠在另一个上）、表面颜色不连续、光照不连续（明 / 暗）。从数学上来看，梯度大的位置为边缘。

+ Canny边缘检测算法
  
  可以分为以下5个步骤：

  + 使用高斯滤波器，以平滑图像，滤除噪声。
  + 计算图像中每个像素点的梯度强度和方向。
  + 应用非极大值（Non-Maximum Suppression）抑制，以消除边缘检测带来的杂散响应。
  + 应用双阈值（Double-Threshold）检测来确定真实的和潜在的边缘。
  + 通过抑制孤立的弱边缘最终完成边缘检测。

### 1.3.2. 角点检测（Corner Detection）
#### 1.3.2.1. Harris角点检测
  
角点检测最原始的想法就是取某个像素的一个邻域窗口（w），当这个窗口在各个方向上进行小范围移动[u,v]时，观察窗口内平均的像素灰度值的变化:

<div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=E(u,v)=\sum_{x,y}w(x,y)[I(x&plus;u,y&plus;v)-I(x,y)]^2" target="_blank"><img src="https://latex.codecogs.com/png.latex?E(u,v)=\sum_{x,y}w(x,y)[I(x&plus;u,y&plus;v)-I(x,y)]^2" title="E(u,v)=\sum_{x,y}w(x,y)[I(x+u,y+v)-I(x,y)]^2" /></a></div>
利用泰勒展开上式可以化为：

<div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=E(u,v)=(u,v)M\binom{u}{v}" target="_blank"><img src="https://latex.codecogs.com/png.latex?E(u,v)=(u,v)M\binom{u}{v}" title="E(u,v)=(u,v)M\binom{u}{v}" /></a></div>

其中结构张量：

<div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=M=\sum_{x,y}w(x,y)\begin{bmatrix}&space;I_xI_x&space;&&space;I_xI_y\\&space;I_xI_y&&space;I_yI_y&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/png.latex?M=\sum_{x,y}w(x,y)\begin{bmatrix}&space;I_xI_x&space;&&space;I_xI_y\\&space;I_xI_y&&space;I_yI_y&space;\end{bmatrix}" title="M=\sum_{x,y}w(x,y)\begin{bmatrix} I_xI_x & I_xI_y\\ I_xI_y& I_yI_y \end{bmatrix}" /></a></div>

通过判断M的两个特征值λ1,λ2的情况，就可以区分出‘flat’，‘edge’，‘corner’这三种区域：

* 角点: λ1,λ2都比较大，数值相近，说明E在每个方向都增大。
* 边缘：λ1远大于λ2，或者λ2远大于λ1，说明E仅在一个方向上增加。
* 平坦区：λ1,λ2都很小，说明E在每个方向上几乎都不变。

还可以通过角点响应函数θ，来判断角点:

  <div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=\theta=det(M)-\alpha&space;trace(M)^2" target="_blank"><img src="https://latex.codecogs.com/png.latex?\theta=det(M)-\alpha&space;trace(M)^2" title="\theta=det(M)-\alpha trace(M)^2" /></a></div>
  
其中α是个常数，属于0.04-0.06。
* 角点：R为大数值整数。
* 边缘：R为大数值负数。
* 平坦区：绝对值R是小数值。

缺陷：图像尺度会影响结果。

#### 1.3.2.2. Shi-Tomasi 算法
Shi-Tomasi 算法是Harris 算法的改进：若两个特征值中较小的一个大于最小阈值，则会得到强角点。
#### 1.3.2.3. SUSAN算法

SUSAN 算子的模板与常规卷积算法的正方形模板不同, 它采用一种近似圆形的模板, 用圆形模板在图像上移动, 模板内部每个图像像素点的灰度值都和模板中心像素的灰度值作比较, 若模板内某个像素的灰度与模板中心像素(核)灰度的差值小于一定值, 则认为该点与核具有相同(或相近)的灰度。由满足这一条件的像素组成的区域称为吸收核同值区(Univalue Segment Assimilating Nucleus, USAN)。



#### 1.3.2.4. FAST算法
过程：
1. 初步筛选，如下图所示，将像素点P1、P5、P9、P13的像素值与中心像素点P的像素值进行比较,
 如果至少有三个像素点的像素值都大于Ip + t，或者都小于Ip - t，则像素点P可能是角点，并在步骤2中进一步判断，否则像素点P不是角点。
 
2. 进一步判断，将像素点P周围的16个像素点的像素值与像素点P的像素值进行比较, 如果有连续12个像素点的像素值都大于Ip +t，或者都小于Ip -
 t，则像素点P是角点。

3. 非极大抑制，首先计算角点的FAST得分（记为V），也就是上一步中12个连续像素点的像素值与该点像素值的差值的绝对值之和，按如下公式计算；然后，对于相邻的两个角点，比较它们的FAST得分，保留得分较大的角点。
   
<div align=center> <img src = 'images/FAST.png'  width="60%"> </img> </div>

<div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=V=max\left\{\begin{matrix}&space;\sum(pixel\&space;values&space;-p)&space;&&space;\&space;\&space;\&space;if(value&space;-p)>t\\&space;\sum(p-pixel\&space;values)&\&space;\&space;\&space;\&space;if(p-value)>t)&space;\end{matrix}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?V=max\left\{\begin{matrix}&space;\sum(pixel\&space;values&space;-p)&space;&&space;\&space;\&space;\&space;if(value&space;-p)>t\\&space;\sum(p-pixel\&space;values)&\&space;\&space;\&space;\&space;if(p-value)>t)&space;\end{matrix}\right." title="V=max\left\{\begin{matrix} \sum(pixel\ values -p) & \ \ \ if(value -p)>t\\ \sum(p-pixel\ values)&\ \ \ \ if(p-value)>t) \end{matrix}\right." /></a></div>

缺陷：

1. 检测效果依赖于阈值t。
   
2. 当图像中存在噪声点时，检测效果不理想。

3. 不产生多尺度特征，所以不具有尺度不变性。
   
4. 特征点没有方向性，这样会失去旋转不变性。
   

### 1.3.3. 识别兴趣点
SIFT检测器（Scale Invariant Feature Transform 尺度不变特征转换）
Sfit算法的实质是在不同的尺度空间上查找关键点（特征点），计算关键点的大小、方向、尺度信息，利用这些信息组成关键点对特征点进行描述的问题。Sift所查找的关键点都是一些十分突出，不会因光照，仿射便函和噪声等因素而变换的“稳定”特征点，如角点、边缘点、暗区的亮点以及亮区的暗点等。

## 1.4. 任务

### 1.4.1. 物体识别/分类

图像分类，该任务需要我们对出现在某幅图像中的物体做标注。比如一共有1000个物体类，对一幅图中所有物体来说，某个物体要么有，要么没有。可实现：输入一幅测试图片，输出该图片中物体类别的候选集。



### 1.4.2. 目标检测

物体检测，包含两个问题，一是判断属于某个特定类的物体是否出现在图中；二是对该物体定位，定位常用表征就是物体的边界框。可实现：输入测试图片，输出检测到的物体类别和位置。

主流算法，分为两类：two-stage，one-stage。

two-stage:先由算法生成一系列作为样本的候选框proposal，再通过卷积神经网络进行样本分类。如Fast R-CNN。

one-stage:不需要region proposal阶段，直接产生物体的类别概率和位置坐标值，经过单次检测即可直接得到最终的检测结果，因此有着更快的检测速度，比较典型的算法如YOLO，SSD，Retina-Net。

### 1.4.3. 图像分割

图像分割是一项稍微复杂的任务，其目的是将图像的各个像素映射到其对应的各个类别。



## 1.5. 需要的知识
1. 数学：概率论，微积分，线性代数，凸优化
2. 编程语言：C++，python
3. 图像处理基础
4. 机器学习，深度学习神经网络，以及开源框架
5. 技能树：

<div align=center><img src = 'images/CVSkillsTree.png'  width="80%"></img></div>



## 1.6. 参考资料
### 1.6.1. C231n：面向视觉识别的卷积神经网络

* [CS231n: Convolutional Neural Networks for Visual Recognition](http://vision.stanford.edu/teaching/cs231n/index.html)

* [CS231n notes](https://cs231n.github.io/)

* [notes中文翻译](https://zhuanlan.zhihu.com/p/34313370)

### 1.6.2. CS31：计算机视觉基础与应用

* [CS131 Computer Vision:  Foundations and Applications](http://vision.stanford.edu/teaching/cs131_fall1819/)
* [CS131_notes](https://github.com/StanfordVL/CS131_notes)
* [notes中文翻译](https://blog.csdn.net/bear507/article/category/7999729)


### 1.6.3. 博客
+ [边缘检测之Canny](https://www.cnblogs.com/techyan1990/p/7291771.html)
+ [图像卷积和滤波的一些知识点](https://blog.csdn.net/zouxy09/article/details/49080029)
## VGGNet

### 简介

VGGNet 由牛津大学的视觉几何组提出，并取得了ILSVRC-2012竞赛定位第一名和分类第二名的成绩。其突出贡献在于证明使用很小的卷积（$3*3$），增加网络深度可以有效提升模型的效果。

### 结构

VGGNet的训练数据是224*224的RGB图片，所做的数据处理就是在样本像素矩阵中减去训练集的均值。采用了大小为3的卷积核（步长为1），大小为2的池化层（步长为1）。VGGNet的网络结构如下：

![](https://github.com/pzxbjx/paper/raw/master/network/_figs/VGGNet/architecture.PNG)

网络深度随ABCDEF递增，加深的网络中前四层卷积层以及后三层全连接层都会用网络A中的参数进行初始化，来加快模型的收敛速度。

为了减少模型中的参数，VGGNet用3个串联的$3*3$卷积核代替之前$7*7$卷积核：

![](https://github.com/pzxbjx/paper/raw/master/network/_figs/VGGNet/kernel.PNG)

### 分类框架

优化算法使用批随机梯度下降，batch size设置为256，momentum为0.9，$L_2$惩罚因子设为$5*10^{-4}$，初始学习率为0.01。

为了获取224*224的输入图片，在每张图片执行SGD迭代时进行一次裁剪，并对图片进行水平翻转以及进行随机RGB色差调整，达到数据增强的目的。对原始图片进行裁剪时，原始图片最小边长过小，裁剪出来的样例会覆盖整个图片，数据增强就失去了意义；若最小边长过大，样例图片只会包含图片中object的一小部分，效果也不好。

论文中提出了两种解决方案：
* 固定原始图片尺寸为256或384
* 随机从[256,512]中的范围里取样，然后将多个不同尺寸的裁剪图片合在一起训练

测试的图片不一定与训练图片尺寸相同，而且不需要裁剪。VGG同样采用了multi-scale的方式，把测试图片放缩到一个尺寸Q，把VGG网络中的全连接层转换成卷积层，通过滑窗的方式得到分类得分图，然后进行平均池化得到分类结果，最后把不同窗口（Q）的结果综合起来就是多尺度预测的结果。

### 实验

![](https://github.com/pzxbjx/paper/raw/master/network/_figs/VGGNet/result1.PNG)

![](https://github.com/pzxbjx/paper/raw/master/network/_figs/VGGNet/result2.PNG)

从上图可以看出来，多尺寸的预测结果稍好于单一尺寸。

当取训练图片S利用尺度抖动的方法范围为[256;512]，测试图片也利用尺度抖动取256，384，512三个值进行分类结果平均值，然后探究对测试图片进行多裁剪估计的方法，即对三个尺度上每个尺度进行50次裁剪（5x5大小的正常网格，并进行两次翻转）即总共150次裁剪的效果图

![](https://github.com/pzxbjx/paper/raw/master/network/_figs/VGGNet/result3.PNG)


### 参考资料
[1.]VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION Karen et al.
[2.]https://blog.csdn.net/qq_31531635/article/details/71170861

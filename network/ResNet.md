## ResNet

### 简介

ResNet（残差神经网络）是由何凯明所在的微软团队提出的一种神经网络框架，通过在网络中引入残差函数，有效解决了深层网络精度饱和的问题，从而进一步加深神经网络的层次以提升模型的效果。

### 残差学习

残差网络发明之前，VGGNet证明了网络的加深可以提升模型的精度，但是网络层数到达一定程度后模型预测的准确率不升反降

![](https://github.com/pzxbjx/paper/raw/master/network/_figs/ResNet/question.PNG)

由上图可知，56层神经网络训练集中错误率高于20层的模型，所以排除了是模型过拟合的可能性。而这个“退化”问题的产生归结于模型的优化难题，常用的各种SGD方法达不到好的学习效果。

针对这个问题，作者提出了一个深度残差学习框架：

![](https://github.com/pzxbjx/paper/raw/master/network/_figs/ResNet/block.PNG)

假设在原本的复数层非线性神经网络中我们想要学习到的是从$x$的一个特征映射$H(x)$，这等价于我们可以学习到一个残差函数，i.e.，$H(x)-x$（假设两者维度相同）。所以与其直接学习特征映射$H(x)$，不如通过这几层网络学习一个残差函数：$F(x)=H(x)-x$，即把原来的$H(x)$转换成了$F(x)+x$。虽然两种表达效果相同，学习优化的难度却不同。作者就是基于残差函数的优化更容易这一点改进了原始的卷积神经网络。

### 网路结构

ResNet的具体网路结构如下：

![](https://github.com/pzxbjx/paper/raw/master/network/_figs/ResNet/framework.PNG)

作者在VGGNet的基础上，进行了一些改进，加入短路（shortcut）机制实现残差单元，直接通过步长为2的卷积核作下采样，用global average代替最后的全连接层。ResNet的设计主要遵循了以下两个原则：
* feature map大小不变时，对应层的滤波器数目不变
* feature map减半时，filter数翻倍以保持计算复杂度不变

由于ResNet的特性，网络层数理论上可以继续叠加下去，论文中给出了不同深度下ResNet的结构：

![](https://github.com/pzxbjx/paper/raw/master/network/_figs/ResNet/architecture.PNG)

从上图我们可以看出，18层和34层的ResNet都是进行两层的残差学习，网络层数更深时，三层的残差单元卷积核大小变为$1 * 1, 3 * 3, 1 * 1$。

![](https://github.com/pzxbjx/paper/raw/master/network/_figs/ResNet/unit.PNG)

而对于短路连接，如果输入输出维度相同，直接element-wise相加即可，如果维度不一致，就不能直接相加，在实验部分，作者采取三种策略：
* 采用恒等映射的方式，维度不一致的时候使用0填充
* 维度一致恒等映射，不一致时使用线性投影保证维度一致
* 统一使用线性投影

### 实验

![](https://github.com/pzxbjx/paper/raw/master/network/_figs/ResNet/training.PNG)

显然，原始的卷积神经网络（plain）观察到明显的“退化”问题，而ResNet有效的解决了这个问题。

![](https://github.com/pzxbjx/paper/raw/master/network/_figs/ResNet/result.PNG)

上图比较了几种短路连接方式下模型的表现，可以看出C>B>A，但是三者的差距比较小，所以说线性投影不是残差网路必须的策略，而A的零填充可以有效减少模型的复杂度，对于网路的加深有明显的帮助。

### 参考资料
[1.]Deep Residual Learning for Image Recognition, Kaiming He et al.
[2.]https://blog.csdn.net/wspba/article/details/56019373
[3.]https://zhuanlan.zhihu.com/p/31852747

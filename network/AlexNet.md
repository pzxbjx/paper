## AlexNet

### 简介

AlexNet 是由多伦多大学Hinton组于2012年提出的深度神经网络结构，通过在神经网络中引入卷积核并采取多种优化策略，AlexNet取得了ILSVRC-2012竞赛分类第一名的成绩。

### 网络结构

![](.\_figs\AlexNet\structure.png)

AlexNet由8层神经网络组成，前五层为卷积层，后三层为全连接层，最后一层是输出为1000维的softmax。

* ReLU

该网络作者Alex Krizhevsky采用ReLU (Rectified Linear Unit，线性整流函数) 作为激活函数，用来代替sigmoid函数和正切函数，可以有效地加快SGD收敛速度。

* Multiple GPUs

神经网络使用两块GTX580显卡进行训练，两块GPU各自训练神经网络的一部分，只有在第二个卷积层和全连接层才互相通信。

* Local Response Normalization

$$
b_{x,y}^i = a_{x,y}^i/(k+\alpha \sum_{j=max(0,i-n/2)}^{min(N-1,i+n/2)}(\alpha_{x,y}^j)^2)^{\beta}
$$

在实际实验中，作者发现采用局部标准化可以提升模型精度，其中$k,n,\alpha,\beta $都是超参数，本文由验证集确定为$k=2,n=5,\alpha=10^{-4},\beta=0.75$。

* Overlapping Pooling

一般的Pooling是不重叠的，而AlexNet使用的Pooling是可重叠的，也就是说，在池化的时候，每次移动的步长小于池化的边长。AlexNet池化的大小为3*3的正方形，每次池化移动步长为2，这样就会出现重叠。从实验结果上来看，重叠池化可以在一定程度上缓解过拟合问题。

### 缓解过拟合

* Data Augmentation（数据增强）

第一种方法是：原始图片大小为256*256，在图片上随机选取224*224的小块进行训练，还可以这些小块进行水平翻转进一步增加数据量；另一种方法是使用PCA改变训练图像RGB通道的像素值。

* Dropout

Dropout是一种有效缓解过拟合的策略，它以一定的概率将神经元置零（通常为0.5），Dropout的作用可以从以下几个角度来解读，一是模拟神经生物学中完成特定任务时，大部分神经元处于失活状态；二是随机使一部分神经元失活，不同层网络组合起来体现了模型集成的思想；三是dropout也可以看做是一种特殊的L1正则

### 训练

![](.\_figs\AlexNet\learning.png)

使用SGD进行训练，batch大小为128，momentum为0.9，weight decay为0.0005，在两块GPU上训练了六天时间

### 实验结果

![](.\_figs\AlexNet\result.png)

在ILSVRC-2010上，top-1错误率为37.5%,top-5错误率为17.0%，上面三排是GPU1学习出来的特征，下半部分是GPU2学习出来的特征。可以明显的看出GPU1学习的特征大多是没有颜色的，两者都通过卷积核学习到了很多关于频率，方向方面的特征。最后一层4096维的向量可用于迁移学习，供其他任务使用。


### 参考资料
[1.]ImageNet Classification with Deep Convolutional Neural Networks, Alex et al.
[2.]https://blog.csdn.net/tinyzhao/article/details/53035944?locationNum=9&fps=1

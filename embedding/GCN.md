##GCN##

###简介###

现实生活中的很多数据集都是呈现出像社交网络，知识网路之类的拓扑图结构，对于这样的图结构，GCN提供了一种应用神经网络模型的方法。

###课题定义###

图$G=(V,E)$，模型输入是$N \times D$的$X$，其中$N$是节点的个数，$D$是节点特征的维度，$A$是图的邻接矩阵,$Z$是模型最后的输出。GCN模型要兼顾图的结构信息和节点的特征，所以我们定义神经网络的forward rule为：
$$
H^{(l+1)}=f(H^{(l)},A)
$$
其中函数$f$代表每层的激活函数，我们有$H^{(0)} = X, H^{(L)} =Z$，

###模型###

我们初定$f = \sigma (AH^{(l)}W^{(l)})$，$W^{(l)}$是对应层网络的参数。

但这个简易的模型有以下两个缺陷：

* 第一个问题是，直接与$A$相乘表示我们把对应节点邻接点的特征信息进行了一个求和，却忽略了该节点自身的特征，因此，我们用$\hat{A} = A + I_n$代替$A$。
* 另一方面是因为矩阵$A$没有进行预处理，直接相乘会改变特征各个维度的尺寸，所以我们对邻接矩阵进行标准化，即$D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$。

所以我们最后得到
$$
f(H^{(l)},A)= \sigma(\hat{D}^{-\frac{1}{2}}\hat{A} \hat{D}^{-\frac{1}{2}} H^{(l)} W^{(l)})
$$
$\sigma$非线性激活函数采用$ReLU$，$\hat{D}$是$\hat{A}$的对角节点度矩阵。

###实验###

在论文$\text{SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS}$中，作者进行了一系列的对比实验检验GCN的鲁棒性。

数据集为：
![](.\_figs\GCN\dataset.png)

GCN模型与其他Baseline算法比较如下：
![](.\_figs\GCN\result.png)

###参考资料###
[1.]SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS, Kipf et al.
[2.]http://tkipf.github.io/graph-convolutional-networks/

ps:上面链接是论文作者写的一篇关于GCN的博客

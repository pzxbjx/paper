## CANE

### 简介

Context-Aware Network Embedding（CANE）是network embedding中的一种模型，在2017年ACL会议上发表。以往的embedding模型，每个网络节点都只有固定的特征表达，论文称之为静态，而CANE对每个网络节点，根据交互的节点不同，学习了不同的表征，所以定义为动态。

### 课题定义

对一个信息网络$G=(V,E,T)$，V表示节点的集合，E表示网络的边，T则对应每个节点的文本信息。每个节点的文本特征可以表示为：$S_v = (w_1,w_2,...,w_{n_v})$，其中$n_v=|S_v|$。网络表征学习的任务就是根据网络的结构，学习一个文本信息的低维嵌入$v \in R^d$。

context-free模型每个节点学习一个静态的表征，context-aware模型每个节点有多少邻接节点就有多少不同的embedding。比如说，对G中的一条边$e_{u,v}$，我们学习的embedding就有$v_{(u)}$和$u_{(v)}$。

节点的表征由两部分组成，结构特征和文本特征，即$v = v_s \oplus v_t$。
我们定义模型的代价函数为：
<img src="http://chart.googleapis.com/chart?cht=tx&chl= f = \sum_{e \in E}L(e)" style="border:none;">

每条边的代价也可以分为两部分，结构代价$L_s(e)$和文本代价$L_t(e)$
$$
L(e) = L_s(e)+L_t(e)
$$
对于$L_s(e)$和$L_t(e)$，我们又有以下的定义：
$$L_s(e)=W_{u,v}logp(v^s|u^s) $$
$$p(v^s|u^s) = \frac{exp(u^s \cdot v^s)}{\sum_{z \in V}exp(u^s \cdot z^s)}$$
$$L_t(e) = \alpha \cdot L_{tt}(e) + \beta \cdot L_{ts}(e) + \gamma \cdot L_{st}(e) $$
$$L_{tt}(e)= W_{u,v}logp(v^t|u^t) $$
$$L_{ts}(e)= W_{u,v}logp(v^t|u^s) $$
$$L_{st}(e)= W_{u,v}logp(v^s|u^t) $$

接下来我们详细介绍文本代价函数的优化。

### context-free

context-free模型的优化主要分为三步
* $Looking-up$：根据初始的word embedding（one-hot），将文本信息转化为文本数值向量
$$
S = (w_1,w_2,...,w_n)
$$
* $convolution$：通过滑窗的形式卷积化，
$$
x_i = C \cdot S_{i:i+l-1}+b
$$
* $max-pooling$：最大池化
$$
r_i = tanh(max(x^i_0,...,x^i_n))
$$
最终我们得到的文本表征为：$v_t = [r_1,...,r_d]^T$，因为$v_t$与交互的节点不相关，所以我们称之为context-free embedding。

### context-aware

context-aware模型流程图如下：
![](https://github.com/pzxbjx/paper/raw/master/embedding/_figs/CANE/model.PNG)

前面流程大体与context-free模型相同，不同的是构建文本表征的时候，作者采用mutual attention的机制，根据交互节点的不同动态分配注意力权重，得到注意力向量：
$$
a^p_i = \frac{exp(g^p_i)}{\sum_{j \in [1,m]} exp(g^p_j)}
$$

### 实验

实验采用了三个数据集
![](https://github.com/pzxbjx/paper/raw/master/embedding/_figs/CANE/dataset.PNG)
在模型训练过程中，为了加快模型的训练速度，作者使用了负采样的trick。
* link prediction
模型表现效果如下：
![](https://github.com/pzxbjx/paper/raw/master/embedding/_figs/CANE/result1.PNG)
![](https://github.com/pzxbjx/paper/raw/master/embedding/_figs/CANE/result2.PNG)
![](https://github.com/pzxbjx/paper/raw/master/embedding/_figs/CANE/result3.PNG)

CANE模型准确率显然要比其他的context-free模型要高。

* vertex classification
对CANE中节点的表征取一个平均池化，我们得到了一个context-aware的embedding，然后用这个embedding进行vertex classification的实验
![](https://github.com/pzxbjx/paper/raw/master/embedding/_figs/CANE/classification.PNG)

从这里我们又可以得到一个结论，CANE可以近似出一个较好的context-free embedding。
最后我们对CANE的embedding作一个可视化操作，直观感受这个模型的优点
![](https://github.com/pzxbjx/paper/raw/master/embedding/_figs/CANE/visual1.PNG)
![](https://github.com/pzxbjx/paper/raw/master/embedding/_figs/CANE/visual2.PNG)

颜色越深的词语表示注意力权重越高，显然，交互节点不同，embedding的侧重点也不同，通过动态的attention机制，CANE很好地挖掘了节点间的相关性

### 参考资料
[1.]CANE: Context-Aware Network Embedding for Relation Modeling, Cunchao Tu et al.

梯度消失问题与如何选择激活函数

**本文结构：**

- 什么是梯度消失？﻿
- 梯度消失有什么影响？
- 是什么原因？
- 解决方案有哪些？
- 如何选择激活函数？
	
---

###1. 什么是梯度消失？﻿

梯度消失，常常发生在用基于梯度的方法训练神经网络的过程中。﻿

当我们在做反向传播，计算损失函数对权重的梯度时，随着越向后传播，梯度变得越来越小，这就意味着**在网络的前面一些层的神经元，会比后面的训练的要慢很多，甚至不会变化。﻿**

---

###2. 有什么影响？

网络的前面的一些层是很重要的，它们负责学习和识别简单的模式，也是整个网络的基础，如果他们的结果不准确的话，那么后面层结果也会**不准确**。﻿

而且用基于梯度的方法训练出参数，主要是通过学习参数的很小的变化对网络的输出值的影响有多大。如果参数的改变，网络的输出值贡献很小，那么就会很难学习参数，**花费时间会非常长**。﻿

---

###3. 梯度消失的原因？

在训练神经网络时，为了让损失函数越来越小，其中一种优化的方法是梯度下降。梯度下降法简单的来说就是在权重的负梯度方向更新权重，如下面这个公式所示，一直到梯度收敛为零。（当然在实际过程中，会通过设定一个超参数叫做最大跌代数来控制，如果迭代次数太小，结果就会不准确，如果迭代次数太大，那么训练过程会非常长。）﻿

![](https://upload-images.jianshu.io/upload_images/1667471-1992fd7bb58c01c6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这里就需要计算参数的梯度，方法是用反向传播。

为了推导一下梯度消失的原因，我们**来看一个最简单的神经网络的反向传播过程**。

![](https://upload-images.jianshu.io/upload_images/1667471-e7baee22ddf815c2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


每个神经元有两个过程，一个是权重与上一层输出的线性组合，一个是作用激活函数。

来看一下最后的损失对第一层权重的梯度是怎样的：

![](https://upload-images.jianshu.io/upload_images/1667471-ed02e59bb6fb39c3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中各部分推导：

![](https://upload-images.jianshu.io/upload_images/1667471-34cc9a0b520a8c99.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

上面用到的激活函数为 sigmoid 函数，黄色曲线为 Sigmoid 的导数，它的值域在 0 到 1/4 之间：

![sigmoid](https://upload-images.jianshu.io/upload_images/1667471-fa2333a372594b42.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

同时一般情况下神经网络在权重初始化时，会按照高斯分布，平均值为0标准差为1这样进行初始化，所以权重矩阵也是小于1的。

于是可以知道：

![](https://upload-images.jianshu.io/upload_images/1667471-1566c70447b9fc9f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

由上面的例子可以看出，**对第一层的权重求的偏导，就有五个小于1的数相乘，那么当层数越多，这就会以指数级迅速减小。**

越靠前的层数，由于离损失越远，梯度计算式中包含的激活函数的导数就越多，那么训练也就越慢。

（那么梯度爆炸，也就是同样的道理，当激活函数的导数大于1的时候，它会呈指数级的增长。）

---

###4. 解决方案有哪些？

由上面的推导我们可以知道，梯度消失的主要原因，主要是和激活函数的导数有关。
所以如果激活函数选择的不合适，就会出现梯度消失问题

当然，除了激活函数，还有其他几种方法：

**梯度消失：**

- 逐层“预训练”（pre-training）＋对整个网络进行“微调”（fine-tunning）
- 选择合适的激活函数
- batch normalization 批规范化：通过对每一层的输出规范为均值和方差一致的方法，消除了 w 带来的放大缩小的影响
- 残差结构
- LSTM

**梯度爆炸：**

- 梯度剪切（ Gradient Clipping）
- 权重正则化
- 选择合适的激活函数
- batch normalization 批规范化，
- RNN 的 truncated Backpropagation through time ，LSTM


今天先来重点看一下激活函数的选择

---

###5. 那么如何选择激活函数呢？通常都有哪些激活函数, 它们的导数长什么样子呢？

由前面的推导可以知道梯度消失的主要原因，是激活函数的导数小于 1，那么在选择激活函数时，就考虑这一点。

有哪些激活函数可以选择呢？

**Relu，**

![relu](https://upload-images.jianshu.io/upload_images/1667471-9e0348f21c467096.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Rectified linear unit，x 大于 0 时，函数值为 x，导数恒为 1，这样在深层网络中使用 relu 激活函数就不会导致梯度消失和爆炸的问题，并且计算速度快。

但是因为 x 小于 0 时函数值恒为0，会导致一些神经元无法激活。


**Leaky Relu，**

![leaky relu](https://upload-images.jianshu.io/upload_images/1667471-bd842dc541230fe7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

是 ReLU 激活函数的变体，为了解决 Relu 函数为 0 部分的问题，当 x 小于 0 时，函数值为 kx，有很小的坡度 k，一般为 0.01，0.02，或者可以作为参数学习而得。

优点
	Leaky ReLU有ReLU的所有优点：计算高效、快速收敛、在正区域内不会饱和
	导数总是不为零，这能减少静默神经元的出现，允许基于梯度的学习
	一定程度上缓解了 dead ReLU 问题

**ELU：**

![elu](https://upload-images.jianshu.io/upload_images/1667471-631c234580ca1e4e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

指数线性单元（Exponential Linear Unit，ELU）也属于 ReLU 的变体。x 小于 0 时为 alpha＊（e^x -1）和其它修正类激活函数不同的是，它包括一个负指数项，从而防止静默神经元出现，导数收敛为零，从而提高学习效率。

优点
	不会有Dead ReLU问题
	输出的均值接近0，zero-centered
缺点
	计算量稍大

**现在最常用的是 Relu，已经成了默认选择，
sigmoid 不要在隐藏层使用了，如果是二分类问题，可以在最后的输出层使用一下，
隐藏层也可以用 tanh，会比 sigmoid 表现好很多。**

此外，下面思维导图总结了其他几种 relu，sigmoid, Tanh 的变体函数，它们的导数，以及优缺点：


![relu 及其变体.jpg](https://upload-images.jianshu.io/upload_images/1667471-6f9fea56650ff972.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![tanh 及其变体.jpg](https://upload-images.jianshu.io/upload_images/1667471-812a41d65968d753.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![sigmoid 及其变体.jpg](https://upload-images.jianshu.io/upload_images/1667471-feed2f5361ecc2d3.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---


学习资料：
http://neuralnetworksanddeeplearning.com/chap5.html
https://dashee87.github.io/data%20science/deep%20learning/visualising-activation-functions-in-neural-networks/
https://blog.csdn.net/qq_25737169/article/details/78847691
https://www.cnblogs.com/willnote/p/6912798.html
https://www.quora.com/What-is-the-vanishing-gradient-problem
https://ayearofai.com/rohan-4-the-vanishing-gradient-problem-ec68f76ffb9b
https://www.learnopencv.com/understanding-activation-functions-in-deep-learning/

---

推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的：
[入门问题][TensorFlow][深度学习][强化学习][神经网络][机器学习][自然语言处理][聊天机器人]


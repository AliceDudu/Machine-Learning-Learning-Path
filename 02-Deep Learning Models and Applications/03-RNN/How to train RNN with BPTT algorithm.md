详解循环神经网络(Recurrent Neural Network)

今天的学习资料是这篇文章，写的非常详细，有理论有代码，本文是补充一些小细节，可以二者结合看效果更好：
https://zybuluo.com/hanbingtao/note/541458


在文末有关于 RNN 的文章汇总，之前写的大多是概览式的模型结构，公式，和一些应用，今天主要放在训练算法的推导。

**本文结构：**

1. 模型
2. 训练算法
3. 基于 RNN 的语言模型例子
4. 代码实现

---

####1. 模型

- 和全连接网络的区别
- 更细致到向量级的连接图
- 为什么循环神经网络可以往前看任意多个输入值


循环神经网络种类繁多，今天只看最基本的循环神经网络，这个基础攻克下来，理解拓展形式也不是问题。

**首先看它和全连接网络的区别：**

下图是一个全连接网络：
它的隐藏层的值只取决于输入的 x

![](http://upload-images.jianshu.io/upload_images/1667471-7d73a2ab30e3353a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

而 RNN 的隐藏层的值 s 不仅仅取决于当前这次的输入 x，还取决于上一次隐藏层的值 s：
这个过程画成简图是这个样子：

![](http://upload-images.jianshu.io/upload_images/1667471-857dd5b7c7015499.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


其中，t 是时刻， x 是输入层， s 是隐藏层， o 是输出层，矩阵 W 就是隐藏层上一次的值作为这一次的输入的权重。

**上面的简图还不能够说明细节，来看一下更细致到向量级的连接图：**

![Elman network](http://upload-images.jianshu.io/upload_images/1667471-4eba217f653527d5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Elman and Jordan networks are also known as "simple recurrent networks" (SRN).

其中各变量含义：
![](http://upload-images.jianshu.io/upload_images/1667471-b05436d24bf2c783.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

输出层是一个全连接层，它的每个节点都和隐藏层的每个节点相连，
隐藏层是循环层。

图来自wiki：https://en.wikipedia.org/wiki/Recurrent_neural_network#Gated_recurrent_unit

**为什么循环神经网络可以往前看任意多个输入值呢？**

来看下面的公式，即 RNN 的输出层 o 和 隐藏层 s 的计算方法：
![](http://upload-images.jianshu.io/upload_images/1667471-55cef3bda3b88ee9.png)

如果反复把式 2 带入到式 1，将得到：
![](http://upload-images.jianshu.io/upload_images/1667471-a3efd4e7588c38fe.png)

这就是原因。

---

####2. 训练算法

**RNN 的训练算法为：BPTT**

BPTT 的基本原理和 BP 算法是一样的，同样是三步：

- 1. 前向计算每个神经元的输出值；
- 2. 反向计算每个神经元的误差项值，它是误差函数E对神经元j的加权输入的偏导数；
- 3. 计算每个权重的梯度。

最后再用随机梯度下降算法更新权重。

BP 算法的详细推导可以看这篇：
[手写，纯享版反向传播算法公式推导](http://www.jianshu.com/p/9e217cfd8a49)
http://www.jianshu.com/p/9e217cfd8a49

**下面详细解析各步骤：**

#####1. 前向计算

计算隐藏层 S 以及它的矩阵形式：
注意下图中，各变量的维度，标在右下角了，
s 的上标代表时刻，下标代表这个向量的第几个元素。

![1](http://upload-images.jianshu.io/upload_images/1667471-016891bce34c7d5a.JPG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


#####2. 误差项的计算

**BTPP 算法就是将第 l 层 t 时刻的误差值沿两个方向传播：**

- 一个方向是，传递到上一层网络，这部分只和权重矩阵 U 有关；（就相当于把全连接网络旋转90度来看）
- 另一个是方向是，沿时间线传递到初始时刻，这部分只和权重矩阵 W 有关。

如下图所示：


![](http://upload-images.jianshu.io/upload_images/1667471-a4ab01fca45b151c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


**所以，就是要求这两个方向的误差项的公式：**

**学习资料中式 3 就是将误差项沿时间反向传播的算法，求到了任意时刻k的误差项**

![](http://upload-images.jianshu.io/upload_images/1667471-c54229027876083e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下面是具体的推导过程：
主要就是用了 链锁反应 和 Jacobian 矩阵

![2](http://upload-images.jianshu.io/upload_images/1667471-c5b0fe716b849cd9.JPG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中 s 和 net 的关系如下，有助于理解求导公式：

![](http://upload-images.jianshu.io/upload_images/1667471-3635524cc19d983e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

 ---

**学习资料中式 4 就是将误差项传递到上一层算法：**

![](http://upload-images.jianshu.io/upload_images/1667471-7b245d2a57a9fac2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这一步和普通的全连接层的算法是完全一样的，具体的推导过程如下：

![3](http://upload-images.jianshu.io/upload_images/1667471-6f1d7f8f03dc8612.JPG)

其中 net 的 l 层 和 l－1 层的关系如下：

![](http://upload-images.jianshu.io/upload_images/1667471-26fa6a2b9f92058b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


---

BPTT 算法的最后一步：计算每个权重的梯度
**学习资料中式 6 就是计算循环层权重矩阵 W 的梯度的公式：**

![](http://upload-images.jianshu.io/upload_images/1667471-cfad00c1614c75eb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


具体的推导过程如下：

![4](http://upload-images.jianshu.io/upload_images/1667471-6af20f2cad55a150.JPG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

**和权重矩阵 W 的梯度计算方式一样，可以得到误差函数在 t 时刻对权重矩阵 U 的梯度：**

![](http://upload-images.jianshu.io/upload_images/1667471-c59ec5feed768ade.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

####3. 基于 RNN 的语言模型例子

我们要用 RNN 做这样一件事情，每输入一个词，循环神经网络就输出截止到目前为止，下一个最可能的词，如下图所示：

![](http://upload-images.jianshu.io/upload_images/1667471-b5f2b0632060338c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



**首先，要把词表达为向量的形式：**

- 建立一个包含所有词的词典，每个词在词典里面有一个唯一的编号。
- 任意一个词都可以用一个N维的one-hot向量来表示。

![](http://upload-images.jianshu.io/upload_images/1667471-723481b390491bda.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这种向量化方法，我们就得到了一个高维、稀疏的向量，这之后需要使用一些降维方法，将高维的稀疏向量转变为低维的稠密向量。

**为了输出 “最可能” 的词，所以需要计算词典中每个词是当前词的下一个词的概率，再选择概率最大的那一个。**

因此，神经网络的输出向量也是一个 N 维向量，向量中的每个元素对应着词典中相应的词是下一个词的概率：

![](http://upload-images.jianshu.io/upload_images/1667471-064119ec5aed9345.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**为了让神经网络输出概率，就要用到 softmax 层作为输出层。**

softmax函数的定义：
因为和概率的特征是一样的，所以可以把它们看做是概率。
![](http://upload-images.jianshu.io/upload_images/1667471-e00d7c18184caf1d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

例：

![](http://upload-images.jianshu.io/upload_images/1667471-424a32ee4086cf54.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


计算过程为：
![](http://upload-images.jianshu.io/upload_images/1667471-b8759ba86b28e09c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


含义就是：
模型预测下一个词是词典中第一个词的概率是 0.03，是词典中第二个词的概率是 0.09。


**语言模型如何训练？**

把语料转换成语言模型的训练数据集，即对输入 x 和标签 y 进行向量化，y 也是一个 one-hot 向量

![](http://upload-images.jianshu.io/upload_images/1667471-b649dd16e9903018.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


**接下来，对概率进行建模，一般用交叉熵误差函数作为优化目标。**

交叉熵误差函数，其定义如下：

![](http://upload-images.jianshu.io/upload_images/1667471-58dfc53c7028ce0b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


用上面例子就是：
![](http://upload-images.jianshu.io/upload_images/1667471-cf0c7933d2135272.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

计算过程如下：
![](http://upload-images.jianshu.io/upload_images/1667471-51e7258b803a986f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


**有了模型，优化目标，梯度表达式，就可以用梯度下降算法进行训练了。**

---

####4. 代码实现

RNN 的 Python 实现代码可以在学习资料中找到。

---


**关于神经网络，写过的文章汇总：**


|  Neural Networks     | Are           | Cool  |
| :------------- |-------------:| -----:|
| 理论      |       |     |
|     | [神经网络的前世](http://www.jianshu.com/p/3a22e8283cda) |   |
|       |       |   [神经网络 之 感知器的概念和实现](http://www.jianshu.com/p/0de1c6723bc9) |
|  |        |    [神经网络 之 线性单元](http://www.jianshu.com/p/af67ad280050) |
|      |   | [手写，纯享版反向传播算法公式推导](http://www.jianshu.com/p/9e217cfd8a49) |
|       |       |   [常用激活函数比较](http://www.jianshu.com/p/22d9720dbf1a) |
|       |       |   [什么是神经网络](http://www.jianshu.com/p/d161a22a0292) |
| 模型 |        |      |
|      | [图解何为CNN](http://www.jianshu.com/p/6daa1af1cf37) |   |
|       |      |  [用 Tensorflow 建立 CNN](http://www.jianshu.com/p/e2f62043d02b) |
|   | [图解RNN](http://www.jianshu.com/p/6c2925ef47f3)     |     |
|       |   | [CS224d－Day 5: RNN快速入门](http://www.jianshu.com/p/bf9ddfb21b07) |
|      |      |   [用深度神经网络处理NER命名实体识别问题](http://www.jianshu.com/p/581832f2c458) |
|   |     |    [用 RNN 训练语言模型生成文本](http://www.jianshu.com/p/b4c5ff7c450f) |
|       |   |  [RNN与机器翻译](http://www.jianshu.com/p/23b46605857e) |
|        |       |   [用 Recursive Neural Networks 得到分析树](http://www.jianshu.com/p/403665b55cd4) |
|   |       |     [RNN的高级应用](http://www.jianshu.com/p/0e840f92b532) |
| TensorFlow     |   |   |
|      | [一文学会用 Tensorflow 搭建神经网络](http://www.jianshu.com/p/e112012a4b2d)      |    |
|       |      |  [用 Tensorflow 建立 CNN](http://www.jianshu.com/p/e2f62043d02b) |
|   |       |   [对比学习用 Keras 搭建 CNN RNN 等常用神经网络](http://www.jianshu.com/p/9efae7a20493) |

---
推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
也许可以找到你想要的
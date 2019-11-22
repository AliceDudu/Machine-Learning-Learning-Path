详解 LSTM

今天的内容有：

1. LSTM 思路
2. LSTM 的前向计算
3. LSTM 的反向传播
4. 关于调参

---

## LSTM 

**长短时记忆网络(Long Short Term Memory Network, LSTM)**，是一种改进之后的循环神经网络，可以解决RNN无法处理长距离的依赖的问题，目前比较流行。

#### 长短时记忆网络的思路：

原始 RNN 的隐藏层只有一个状态，即h，它对于短期的输入非常敏感。
再增加一个状态，即c，让它来保存长期的状态，称为单元状态(cell state)。

![](http://upload-images.jianshu.io/upload_images/1667471-c9dbab3979794684.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

把上图按照时间维度展开：

![](http://upload-images.jianshu.io/upload_images/1667471-38098a4880e6d5ee.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


在 t 时刻，LSTM 的输入有三个：当前时刻网络的输入值 `x_t`、上一时刻 LSTM 的输出值 `h_t-1`、以及上一时刻的单元状态 `c_t-1`；
LSTM 的输出有两个：当前时刻 LSTM 输出值 `h_t`、和当前时刻的单元状态 `c_t`.

#### 关键问题是：怎样控制长期状态 c ？
方法是：使用三个控制开关

![](http://upload-images.jianshu.io/upload_images/1667471-1cfe33538b6c87cd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

第一个开关，负责控制继续保存长期状态c；
第二个开关，负责控制把即时状态输入到长期状态c；
第三个开关，负责控制是否把长期状态c作为当前的LSTM的输出。

#### 如何在算法中实现这三个开关？
方法：用 门（gate）

定义：gate 实际上就是一层全连接层，输入是一个向量，输出是一个 0到1 之间的实数向量。
公式为：
![](http://upload-images.jianshu.io/upload_images/1667471-bb8d1d0c2c5aa0f2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

回忆一下它的样子：
![](http://upload-images.jianshu.io/upload_images/1667471-7901f46a0180b280.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### gate 如何进行控制？
方法：用门的输出向量按元素乘以我们需要控制的那个向量
原理：门的输出是 0到1 之间的实数向量，
当门输出为 0 时，任何向量与之相乘都会得到 0 向量，这就相当于什么都不能通过；
输出为 1 时，任何向量与之相乘都不会有任何改变，这就相当于什么都可以通过。


---

## LSTM 前向计算

在 LSTM－1 中提到了，模型是通过使用三个控制开关来控制长期状态 c 的：

![](http://upload-images.jianshu.io/upload_images/1667471-1cfe33538b6c87cd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这些开关就是用门（gate）来实现：

![](http://upload-images.jianshu.io/upload_images/1667471-7901f46a0180b280.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

接下来具体看这三重门

---

### LSTM 的前向计算:

一共有 6 个公式

**遗忘门（forget gate）**
它决定了上一时刻的单元状态 `c_t-1` 有多少保留到当前时刻 `c_t`

**输入门（input gate）**
它决定了当前时刻网络的输入 `x_t` 有多少保存到单元状态 `c_t`

**输出门（output gate）**
控制单元状态 `c_t` 有多少输出到 LSTM 的当前输出值 `h_t`

---

#### 遗忘门的计算为：

![forget](http://upload-images.jianshu.io/upload_images/1667471-81bc580ebc1b51e9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

遗忘门的计算公式中：
`W_f` 是遗忘门的权重矩阵，`[h_t-1, x_t]` 表示把两个向量连接成一个更长的向量，`b_f` 是遗忘门的偏置项，`σ` 是 sigmoid 函数。

---

#### 输入门的计算：


![input](http://upload-images.jianshu.io/upload_images/1667471-f013288618c83b31.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

根据上一次的输出和本次输入来计算当前输入的单元状态：

![当前输入的单元状态c_t](http://upload-images.jianshu.io/upload_images/1667471-6443c16af1fc9fa1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

当前时刻的单元状态 `c_t` 的计算：由上一次的单元状态 `c_t-1` 按元素乘以遗忘门 `f_t`，再用当前输入的单元状态 `c_t` 按元素乘以输入门 `i_t`，再将两个积加和：
这样，就可以把当前的记忆 `c_t` 和长期的记忆 `c_t-1` 组合在一起，形成了新的单元状态 `c_t`。
由于遗忘门的控制，它可以保存很久很久之前的信息，由于输入门的控制，它又可以避免当前无关紧要的内容进入记忆。

![当前时刻的单元状态c_t](http://upload-images.jianshu.io/upload_images/1667471-9addd095ca99f567.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

#### 输出门的计算：

![output](http://upload-images.jianshu.io/upload_images/1667471-963ff8645885f284.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




---


## LSTM 的反向传播训练算法

主要有三步：

**1. 前向计算每个神经元的输出值，一共有 5 个变量，计算方法就是前一部分：**

![](http://upload-images.jianshu.io/upload_images/1667471-e7209fdb040ea1da.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**2. 反向计算每个神经元的误差项值。与 RNN 一样，LSTM 误差项的反向传播也是包括两个方向：**
一个是沿时间的反向传播，即从当前 t 时刻开始，计算每个时刻的误差项；
一个是将误差项向上一层传播。

**3. 根据相应的误差项，计算每个权重的梯度。**

---

gate 的激活函数定义为 sigmoid 函数，输出的激活函数为 tanh 函数，导数分别为：

![](http://upload-images.jianshu.io/upload_images/1667471-a91de874e723e790.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

具体推导公式为：

![](http://upload-images.jianshu.io/upload_images/1667471-33c068c8e5a4b235.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![](http://upload-images.jianshu.io/upload_images/1667471-7701441080cb29b7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

具体推导公式为：

![](http://upload-images.jianshu.io/upload_images/1667471-c6e64fa1e9822b32.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

**目标是要学习 8 组参数，如下图所示：**



![](http://upload-images.jianshu.io/upload_images/1667471-9da34d2b2b475e7a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


又权重矩阵 W 都是由两个矩阵拼接而成，这两部分在反向传播中使用不同的公式，因此在后续的推导中，权重矩阵也要被写为分开的两个矩阵。


接着就来求两个方向的误差，和一个梯度计算。
这个公式推导过程在本文的学习资料中有比较详细的介绍，大家可以去看原文：
https://zybuluo.com/hanbingtao/note/581764

---

#### 1. 误差项沿时间的反向传递：

定义 t 时刻的误差项：

![](http://upload-images.jianshu.io/upload_images/1667471-36b329f0e74bc182.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

目的是要计算出 t-1 时刻的误差项：

![](http://upload-images.jianshu.io/upload_images/1667471-05a6a2d7bf14706d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


利用 h_t c_t 的定义，和全导数公式，可以得到 将误差项向前传递到任意k时刻的公式：

![](http://upload-images.jianshu.io/upload_images/1667471-fd64d232f4e83965.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

#### 2. 将误差项传递到上一层的公式：


![](http://upload-images.jianshu.io/upload_images/1667471-48506e81bde6ac35.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---


#### 3. 权重梯度的计算：


![](http://upload-images.jianshu.io/upload_images/1667471-229f884af6e9ef36.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

以上就是 LSTM 的训练算法的全部公式。

---

#### 关于它的 Tuning 有下面几个建议：

![](http://upload-images.jianshu.io/upload_images/1667471-a2e70323d48bd259.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

来自 LSTM Hyperparameter Tuning：
https://deeplearning4j.org/lstm


#### 还有一个用 LSTM 做 text_generation 的例子

https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py



学习资料：
https://zybuluo.com/hanbingtao/note/581764

---

推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的

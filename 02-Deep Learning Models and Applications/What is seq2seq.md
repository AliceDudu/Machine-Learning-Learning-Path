seq2seq 入门

本文结构：

- 什么是 seq2seq？
- Encoder–Decoder 结构？
- seq2seq 结构？

---

#### 什么是 seq2seq？

seq2seq 是一个 **Encoder–Decoder 结构**的网络，它的输入是一个序列，输出也是一个序列， Encoder 中将一个可变长度的信号序列变为固定长度的向量表达，Decoder 将这个固定长度的向量变成可变长度的目标的信号序列。

这个结构**最重要的地方**在于输入序列和输出序列的长度是可变的，可以用于翻译，聊天机器人，句法分析，文本摘要等。

下面是写过的 seq2seq 的应用：

RNN与机器翻译
http://www.jianshu.com/p/23b46605857e
如何自动生成文本摘要
http://www.jianshu.com/p/abc7e13abc21
自己动手写个聊天机器人吧
http://www.jianshu.com/p/d0f4a751012b

---

#### Encoder–Decoder 结构？

**Cho 在 2014 年提出了 Encoder–Decoder 结构**，即由两个 RNN 组成，
https://arxiv.org/pdf/1406.1078.pdf

![](http://upload-images.jianshu.io/upload_images/1667471-016a0979117ebe7d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

（其中的 RNNCell 可以用 RNN ，GRU，LSTM 等结构）

**在每个时刻， Encoder** 中输入一个字/词，隐藏层就会根据这个公式而改变，
![](http://upload-images.jianshu.io/upload_images/1667471-4a9452b442a61fc7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

到最后一个字/词 XT 时 ，隐藏层输出 c ，因为 RNN 的特点就是把前面每一步的输入信息都考虑进来了，所以 **c 相当于把整个句子的信息都包含了**，可以看成整个句子的一个语义表示。

**Decoder 在 t 时刻**的隐藏层状态 ht 由 ht−1，yt−1，c 决定：
![](http://upload-images.jianshu.io/upload_images/1667471-5538b5486dd4b754.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

yt 是由 ht，yt−1，c 决定：
![](http://upload-images.jianshu.io/upload_images/1667471-eec4bce3474f038f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

f 和 g 都是激活函数，其中 g 函数一般是 softmax。

模型最终是要最大化下面这个对数似然条件概率：

![](http://upload-images.jianshu.io/upload_images/1667471-00f8786ca402cbd3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中每个  (xn, yn) 表示一对输入输出的序列， θ 为模型的参数。

---

#### seq2seq 结构？

Sutskever 在 2014 年也发表了论文：
https://arxiv.org/pdf/1409.3215.pdf

**这个模型结构更简单，**

![](http://upload-images.jianshu.io/upload_images/1667471-458e7ba0fbeaa902.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

因为 **Decoder 在 t 时刻** yt 是由 ht，yt−1 决定，而没有 c：
![](http://upload-images.jianshu.io/upload_images/1667471-9054dfa5318163fd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

论文中的 **Encoder 和 Decoder 都用的 LSTM 结构**，注意每句话的末尾要有  “<EOS>” 标志。 Encoder 最后一个时刻的状态 [cXT,hXT] 就和第一篇论文中说的中间语义向量 c 一样，它将作为 Decoder 的初始状态，在 Decoder 中，每个时刻的输出会作为下一个时刻的输入，直到 Decoder 在某个时刻预测输出特殊符号 <END> 结束。

LSTM 的目的是估计条件概率 p(y1, . . . , yT′ |x1, . . . , xT ) ，
它先通过最后一个隐藏层获得输入序列  (x1, . . . , xT ) 的固定长度的向量表达 v，
然后用 LSTM-LM 公式计算输出序列 y1, . . . , yT′ 的概率，
在这个公式中，初始状态就是 v，

![](http://upload-images.jianshu.io/upload_images/1667471-35ebc854841a743d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**而且用了 4 层的 LSTM，而不是一层：**论文中的实验结果表明深层的要比单层的效果好
下面是个 3 层的例子

![](http://upload-images.jianshu.io/upload_images/1667471-dc52883e89b07014.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

计算公式如下：
![](http://upload-images.jianshu.io/upload_images/1667471-23afe94736d80956.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](http://upload-images.jianshu.io/upload_images/1667471-8db786dbf9972914.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

为了便于理解，单层的表示如下：
![](http://upload-images.jianshu.io/upload_images/1667471-d555778e04c97dd8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


**并且对输入序列做了一个翻转**，即不是把 a, b, c 映射到 α, β, γ, 而是把 c, b, a 映射到 α, β, γ, 这样的结果是相应的 a 会更接近 α，并且更利于  SGD 建立输入输出间的关系。

---

参考：
Learning Phrase Representations using RNN Encoder–Decoder
for Statistical Machine Translation
https://arxiv.org/pdf/1406.1078.pdf

Sequence to Sequence Learning
with Neural Networks
https://arxiv.org/pdf/1409.3215.pdf

Generating Sequences With
Recurrent Neural Networks
https://arxiv.org/pdf/1308.0850.pdf

---
推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的
什么是神经网络 

**本文结构：**

1. 什么是神经网络
2. 什么是神经元
3. 神经网络的计算和训练
4. 代码实现

---

###1. 什么是神经网络

神经网络就是按照一定规则将多个神经元连接起来的网络

例如全连接(full connected, FC)神经网络，它的规则包括：

- 有三种层：输入层，输出层，隐藏层。
- 同一层的神经元之间没有连接。
- full connected的含义：第 N 层的每个神经元和第 N-1 层的所有神经元相连，第 N-1 层神经元的输出就是第 N 层神经元的输入。
- 每个连接都有一个权值。

不同的神经网络，具有不同的连接规则

---

###2. 什么是神经元

神经元和感知器的区别也是在激活函数：
感知器，它的激活函数是阶跃函数，神经元，激活函数往往选择为 sigmoid 函数或 tanh 函数等

![](http://upload-images.jianshu.io/upload_images/1667471-7901f46a0180b280.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中 sigmoid 函数的公式和图表示如下：

![](http://upload-images.jianshu.io/upload_images/1667471-dda19b272dc80835.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![](http://upload-images.jianshu.io/upload_images/1667471-41128206c0129730.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

sigmoid 函数的求导公式：

![](http://upload-images.jianshu.io/upload_images/1667471-38dff7ddc8aca775.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

想了解更多还可以看这篇：[常用激活函数比较](http://www.jianshu.com/p/22d9720dbf1a)

---

###3. 神经网络的训练

![](http://upload-images.jianshu.io/upload_images/1667471-7d73a2ab30e3353a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**先向前计算，再向后传播**

例如上面神经网络的结构

输入层，首先将输入向量的每个元素的值，赋给输入层的对应神经元

隐藏层，前一层传递过来的输入值，加权求和后，再输入到激活函数中，根据如下公式，向前计算这一层的每个神经元的值

![](http://upload-images.jianshu.io/upload_images/1667471-36821a5b4d116cd7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

输出层的计算和隐藏层的一样

![](http://upload-images.jianshu.io/upload_images/1667471-db05203b9f49117a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

用矩阵来表示

![](http://upload-images.jianshu.io/upload_images/1667471-dc2123304d82b4a5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这个公式适用于每个隐藏层和输出层，就是 W 的值和 f 的形式会不一样，
其中 W 是某一层的权重矩阵，x 是某层的输入向量，a 是某层的输出向量



**模型要学习的东西就 W。**

诸如神经网络的连接方式、网络的层数、每层的节点数这些参数，不是学习出来的，而是人为事先设置的，称之为超参数。


训练它们的方法和前面感知器中用到的一样，就是要用梯度下降算法：

![](http://upload-images.jianshu.io/upload_images/1667471-10914f9857dabcda.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**完整的推导可以看这篇，一步一步很详细：**
[手写，纯享版反向传播算法公式推导](http://www.jianshu.com/p/9e217cfd8a49)



part 4. 代码实现 下次再写



学习资料：
https://www.zybuluo.com/hanbingtao/note/476663


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

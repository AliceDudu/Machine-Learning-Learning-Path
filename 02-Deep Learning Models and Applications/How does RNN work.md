CS224d－Day 5: RNN快速入门

---

CS224d－Day 5: 什么是RNN

**本文结构：**

- 1.什么是 RNN？和NN的区别？
- 2.RNN 能做什么？为什么要用 RNN？
- 3.RNN 怎么工作的？
- 4.RNN 基本模型存在某些问题？
- 5.GRU 和 LSTM 是什么？

---

#### 1.什么是 RNN？和NN的区别？

RNN－(Recurrent Neural Networks)：循环神经网络

传统的神经网络模型，它是有向无环的，就是在隐藏层中各个神经元之间是没有联系的，而实际上我们的大脑并不是这样运作的，所以有了RNN模型，它在隐藏层的各个神经元之间是有相互作用的，能够处理那些输入之间前后有关联的问题。

![](http://upload-images.jianshu.io/upload_images/1667471-13c3ef34f72b424e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


#### 2.RNN 能做什么？为什么要用 RNN？

RNN 在 NLP 中有广泛的应用，语言模型与文本生成(Language Modeling and Generating Text)，机器翻译(Machine Translation)，语音识别(Speech Recognition)，图像描述生成 (Generating Image Descriptions) 等。


#### 3.RNN 怎么工作的？
参考：[深入浅出讲解 SRN](http://v.youku.com/v_show/id_XMTI2MzI2Mzg4NA==.html)

用 SRN－(Simple RNNs) 这个最简单的 RNN 模型来举例说明一下它是怎样工作的：

SRN 做的事情就是，在一个时间序列中寻找这个时间序列具有的结构。例如，给一句话，这句话是把所有词串在一起没有空格，然后 SRN 要自动学习最小单元是什么，也就是它要学习哪些是单词，怎样的切割才可以被识别成是一个单词。

具体做法就是，在每个时间点时，预测下一个时间点是什么字母，SRN 的结果就是随着时间预测结果的误差，比如说，t＝0时字母是F，t＝1时预测是i，那误差就会减小，直到d，误差都是一直减小，但是下一刻预测结果是S，误差就会重新升高。就这样通过 SRN 这个模型就可以得到这个时间序列是由哪些词组成的。

![](http://upload-images.jianshu.io/upload_images/1667471-5aa2ebb15cd6621f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




模型结构就是有3层，输入层隐藏层和输出层，另外还有一个语义层，语义层的内容是直接复制上一时刻隐藏层的内容，然后它会返回一个权重矩阵，t 时刻的输入层和由 t－1 隐藏层复制过来的语义层一同作用到 t 时刻的输出层。

![](http://upload-images.jianshu.io/upload_images/1667471-d72d13bc8a343e7d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


再具体点，把这个模型的环拆成线型来理解，在 t＝0 的时候，就是普通的神经网络模型，有3层，两个权重矩阵和 bias，到输出层，预测结果和目标结果计算误差，接着用 BP 去更新 W1 和 W2，但是在 t＝1 的时候，就有一个语义层，是从上一个时刻的隐藏层复制过来的，然后和此刻的输入层一起作用到隐藏层，再继续得到结果，再通过 BP 去更新 W1 和 W2。一直这样下去不断地迭代 W1，W2，theta，不断地跑这个时间序列，如果串的长度不到迭代次数，就首尾相连，直到收敛停止迭代。
![](http://upload-images.jianshu.io/upload_images/1667471-c640fd72c9fc9cb4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


SRN  是由 ELMAN 提出的，他用 N 个词，造了几百个句子，然后首尾相连，放进网络进行学习，最终结果就是学到了里面的基本构成单元－单词。
![](http://upload-images.jianshu.io/upload_images/1667471-72a13da92ecd7ca0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


用数学表达出这个模型：
![](http://upload-images.jianshu.io/upload_images/1667471-b0996b7e2c292d0b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


#### 4.RNN 基本模型存在某些问题？

不过这个模型有个问题，就是当想要记忆的越多时，比如想要记忆 t－1，t－2，t－3 时刻的，就需要更多的层，伴随着层数的增加，就会出现 梯度消失(vanishing gradients) 的问题，

梯度消失就是一定深度的梯度对模型更新[没有帮助。](http://caffecn.cn/?/question/238)

原因简述：更新模型参数的方法是反向求导，越往前梯度越小。而激活函数是 sigmoid 和 tanh 的时候，这两个函数的导数又是在两端都是无限趋近于0的，会使得之前的梯度也朝向0，最终的结果是到达一定”深度“后，梯度就对模型的更新没有任何贡献。

这篇博客中有详细的解释为何会出现[这样的问题。](http://blog.csdn.net/qq_29133371/article/details/51867856)

可以用 gradient clipping 来改善这个问题：

![](http://upload-images.jianshu.io/upload_images/1667471-b6db52ab934f32b6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



#### 5.GRU 和 LSTM 是什么？


**GRU：**


为了解决上面的问题，让 RNN 有更好的表现，它有一些改良版模型。

GRU(Gated Recurrent Unit Recurrent Neural Networks) 

![](http://upload-images.jianshu.io/upload_images/1667471-8d2e057665620930.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![](http://upload-images.jianshu.io/upload_images/1667471-eae2b092b8801947.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


GRU 对两个方面进行了改进：1. 序列中不同的位置的单词对当前的隐藏层的状态的影响不同，越前面的影响越小。2. 误差可能是由某一个或者几个单词引起的，更新权值时应该只针对相应的单词。

**LSTM：**

LSTM (Long Short-Term Memory，长短时记忆模型) 是目前使用最广泛的模型，它能够更好地对长短时依赖进行表达。




![](http://upload-images.jianshu.io/upload_images/1667471-2bab5ecfe5e00af0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![](http://upload-images.jianshu.io/upload_images/1667471-ee5cf04bf174284d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


LSTM 与 GRU 类似，只是在隐藏层使用了不同的函数。这里有一篇非常好的文章来[讲解 LSTM。](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
[简书上也有一篇译文。](http://www.jianshu.com/p/9dc9f41f0b29#)

##### [cs224d]

Day 1. [深度学习与自然语言处理 主要概念一览](http://www.jianshu.com/p/6993edef96e4)
Day 2. [TensorFlow 入门](http://www.jianshu.com/p/6766fbcd43b9)
Day 3. [word2vec 模型思想和代码实现](http://www.jianshu.com/p/86134284fa14)
Day 4. [怎样做情感分析](http://www.jianshu.com/p/1909031bb1f2)
Day 5. [CS224d－Day 5: RNN快速入门](http://www.jianshu.com/p/bf9ddfb21b07)
Day 6. [一文学会用 Tensorflow 搭建神经网络](http://www.jianshu.com/p/e112012a4b2d)
Day 7. [用深度神经网络处理NER命名实体识别问题](http://www.jianshu.com/p/581832f2c458)
Day 8. [用 RNN 训练语言模型生成文本](http://www.jianshu.com/p/b4c5ff7c450f)
Day 9. [RNN与机器翻译](http://www.jianshu.com/p/23b46605857e)
Day 10. [用 Recursive Neural Networks 得到分析树](http://www.jianshu.com/p/403665b55cd4)
Day 11. [RNN的高级应用](http://www.jianshu.com/p/0e840f92b532)

---
我是 *不会停的蜗牛Alice*
85后全职主妇
喜欢人工智能，行动派
创造力，思考力，学习力提升修炼进行中
欢迎您的喜欢，关注和评论！
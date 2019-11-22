RNN与机器翻译

---

CS224d-Day 9:
 GRUs and LSTMs -- for machine translation
[视频链接](https://www.youtube.com/watch?v=qGlmW2n4s1w&index=9&list=PLlJy-eBtNFt4CSVWYqscHDdP58M3zFHIG)
[课件链接](https://cs224d.stanford.edu/lectures/CS224d-Lecture9.pdf)

---

本文结构：
- **机器翻译系统整体的认识**
  - **什么是 parallel corpora**
  - **三个模块**
  - **各模块有什么难点**

- **RNN 模型**
  - **最简单的 RNN 模型**
  - **扩展模型**
    - **GRU:**
    - **LSTM**


---

下面是video的笔记：

### 1.机器翻译
**机器翻译**是NLP问题中比较难的其中之一，为了解决这个问题，有一些很好玩的模型：

- Gated Recurrent Units by Cho et al. (2014)
http://arxiv.org/pdf/1412.3555v1.pdf
http://arxiv.org/pdf/1502.02367v3.pdf

- Long-Short-Term-Memories by Hochreiter and Schmidhuber (1997)
http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf

LSTM 是很早以前的模型，GRU 是比较新的。


过去的方式很传统，现在的 Deep Learning 是基于统计的，它们以 parallel corpora 为基础。

**什么是 parallel corpora？**
是个很大的 corpora，句子和段落都是对齐的.

比如 European	Parliament，欧洲议会 的笔记，它们被欧盟的所有语言所记录，所以你会看到一句英语，法语，意大利语，德语等等。


通常我们只需要一对语言，也就是一句翻译成一句。
此外我们也有翻译整段文字的场景，一样的，这个时候就是段落的对齐。

这是一个非常复杂的系统，先做一个**整体的认识**，然后再看具体的模块：

![](http://upload-images.jianshu.io/upload_images/1667471-7ed0b89617030bc5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

例如输入的语言是 French，目标语言是 English，我们希望翻译的概率达到最大。
也就是，如果给了f，它被翻译成e^的概率最大，那e^就是翻译的结果。
![](http://upload-images.jianshu.io/upload_images/1667471-9761103b7d990bec.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这里有**三个模块:**

第一个是 Translation Model，是通过训练 parallel corpora 得到的，就是有一句 French，一句对应的 English。

第二个是 Language	Model，它只通过 English 来训练 ，这是一个随机的 corporate，我们只是需要一个单语言 English 的语料库，所以可以是任意的 Wikipedia，句子，段落。

第一个模块，会把输入的 French 句子，切割成单词和短语，然后去 parallel corpora 找对应，然后再把它们拼起来。
这个过程中，会得到很多候选者，第二个模块中，就用 Language Model 给它们重新打分，把句子变的通顺。

然后在第三个模块 Decoder 中，将两个模块合起来，它会将所有的翻译进行打分，最终返回一个最合理的结果。

接下来，**具体看：**

第一个 Translation Model，目标是要知道 输入的 French 对应的是什么 English。
这一步很难，CS224n 中有专门讲这一步。

**有什么难点？**
这个图里，左边是 English，右边是 French，右边的 Le 是没有对齐翻译的，模型需要识别出来。


![](http://upload-images.jianshu.io/upload_images/1667471-95ea0d01072ce1d0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


在下面这个例子中，左边的 and 在 parallel corpora 是没有翻译的，还有一种情况是 左边的 implemented 对应着多个翻译。


![](http://upload-images.jianshu.io/upload_images/1667471-eda6fc05f875e968.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这还算简单点，因为 English 和 French 的语序差不多，如果遇到不同语序的一对，那几乎需要翻转所有单词。

还会有一个 French 可以对应多个 English：
![](http://upload-images.jianshu.io/upload_images/1667471-ff24c25276db842e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

还有多个 French 对应多个 English：
![](http://upload-images.jianshu.io/upload_images/1667471-4deac5b2990d13b0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



通过第一个模块后，假设我们已经给输入的语言找到了最有可能的短语结果，接下来想要根据语法知识形成一个完整的句子的翻译，而不只是单词的对应。
![](http://upload-images.jianshu.io/upload_images/1667471-a8593c6b946038e0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**Decoder:**
这里的难点是，有很多可能的选择，很多不同顺序构成的句子。
![](http://upload-images.jianshu.io/upload_images/1667471-568dfba259234399.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们要在所有可能的组合中找到最有可能的结果，这是个庞大的搜索任务。

![](http://upload-images.jianshu.io/upload_images/1667471-e6d0560fc0504f81.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这只是一个简单的概括，机器翻译是个很庞大的系统，由不同的模型组成，分别处理不同的问题，还有很多重要的细节这里都没有讲。


### 2. RNN模型
那么 深度学习 可以简化这个系统吗？**只用一个 RNN 就能做到机器翻译吗？**目前还没有达到这个水平，最新的一篇文章，还没有超过最好的机器翻译系统。不过，这几个并不是翻译专业的作者，用了一年的时间，就可以训练出只比传统最好的模型的准确度低0.5%的模型。

![](http://upload-images.jianshu.io/upload_images/1667471-9bb28e40d7300b66.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

上图是将 德语 翻译成 英语，模型的输入是 词向量，然后经过 recurrent neural network ，用 logistic regression，这一次不是预测下一个单词，而是预测一个完整的句子组成形式。

下面是**最简单的 RNN 模型**，在 Encoder 中，每一个输入的词向量都会经过线性变换，Encoder 是一个 recurrent neural network，Decoder 是同样的 RNN，每一次得到一个最有可能的翻译结果，然后让所有单词的 cross entropy 达到最小。

![](http://upload-images.jianshu.io/upload_images/1667471-1d5e58dfbcb1ccbe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

接下来是一些**扩展模型。**

第一个是训练 encoding 和 decoding 的不同的 weights。
decoder 可以是预先训练好的语言模型，只需要预测合理的 English 短语。
由上面的图里的公式， encoder 和 decoder 的 phi 函数是不一样的。
![](http://upload-images.jianshu.io/upload_images/1667471-f1bd4c27c42823f8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


扩展2是计算 decoder 的每个隐藏层，输入有三部分：上一状态的隐藏层，encoder的最后一个向量，前一个预测出来的单词 y_(t-1)

![](http://upload-images.jianshu.io/upload_images/1667471-b1074ebda7cfbe8d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下面这个图是上图的具体化，意思是一样的，

![](http://upload-images.jianshu.io/upload_images/1667471-28ab2bc0882ad882.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


扩展3是把隐藏层数增加。
扩展4是 bi-directional encoder, 就是不只是用encoder的最后一个向量，还有第一个向量。？？？？没太懂这个模型的意义
![](http://upload-images.jianshu.io/upload_images/1667471-6da4bdd162e24480.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

扩展5是把输入的句子倒置后再去翻译。

### **GRU:**
扩展6是 Better Units，就是GRU，论文：http://arxiv.org/pdf/1412.3555v1.pdf

GRU 为了获取更长的记忆，它想要保持住隐向量的某些元素，所以在下面这个公式里，我们并不想要f内部的内积，因为这个矩阵相乘会改变隐藏层的状态。你可以将保持的记忆信息带到后面的很多步里。

![](http://upload-images.jianshu.io/upload_images/1667471-d27a145287986eeb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

GRU的定义：

标准的RNN在下一步计算隐藏层，GRU有很多recurrent units，再计算最后的h。

前两步是计算两个不同的gate.

第一个是 update gate，这里并没有用随机的非线性函数，只是用了sigmoid，代表门是可以on和off的，开／关的状态下会带来不同的影响，update gate zt 依赖于当前的输入词向量和上一步的隐藏层状态。
![](http://upload-images.jianshu.io/upload_images/1667471-6d2965e1104dda97.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

第二个是 reset gate，输入和 zt 是一样的，不过 W 和 U 矩阵是不一样的。

有了两个 gate，下一步是想要有一个临时的 memory content：h_t tail.
当 reset gate 是0时，那过去的记忆并不重要，就可以完全忽略过去的记忆 h_(t-1)。
**例如做情感识别时**，一句影评说了 awsome，后面接了一段无关紧要的话，那么这些话就可以忽略，就只需要考虑 current input word。

最后的 h_t，当 z_t=1 时，最后的记忆就只是copy上一状态的记忆，而不需要考虑过去的很多记忆。当 z_t=0 时，就需要考虑当前word以及它和前面记忆的联系。

![](http://upload-images.jianshu.io/upload_images/1667471-b74634fa02f87356.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下图是 GRU 的一个**简洁表示图：**

input是词向量，然后计算两个 gate，再计算 reset memory，
当根据 h\_(t-1), h\_t tail 计算 h\_t 时，z\_t 起到修正的作用，当计算 h\_t tail 时，r\_t 起到修正的作用。
也就是，GRU是为了达到一个长期的记忆，而是否需要长期记忆是由 z\_t 控制的，当z\_t＝1时，它把h\_(t-1)复制过来，意味着过去的 梯度，误差等信息，都被复制过来，这在做 back propagation 的时候，误差不会越来越小，只是被传送到后面。

![](http://upload-images.jianshu.io/upload_images/1667471-6ea917a3dd39742d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

GRU还可以用来做count的任务，而标准的RNN就很难做到。

小结：
最后一句，例如，当你的 前一半 reset gate 非常活跃时，那隐藏层的前一半就会持续地更新，而另一半则是0，你不会把它们考虑进去。
![](http://upload-images.jianshu.io/upload_images/1667471-b8052ef971008c63.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

写成代码就是，定义一个 GRU 的class，输入inputs后会经历两个function，forward propagation，back propagation，在forward时计算 gates 等变量。back 函数会把delta加上一些gradient再输出一个新的delta。

这里有一个不错的 Theano－GRU 代码：
[GRU的代码theano：](https://github.com/dennybritz/rnn-tutorial-gru-lstm/blob/master/gru_theano.py)

### **LSTM**
下面是**LSTM**，很早的模型啦。

有三个 gate，

第一个 input gate，当 current cell 很重要的时候，就希望更新它，i_t 的值就很大。
h_(t-1)就代表着 memory。

第二个 forget gate，就是忘记你的过去吧，忘记你至今学到的所有。

第三个 output gate，有了它，你就不需要把内部的状态告诉后面的网络，只需要keep around就可以了，不需要tell。

当你想要预测单词时，当你看到一整句话时，你只需要输出一个结果，并不需要释放所有的隐藏层和记忆。给了两个facts，然后进行推理，再给出output即可。

**例如，在一个问答系统里**，当你问 where is the toy, 你现在只知道 John took the toy, 所以现在你无法回答问题，但是又经过了几步，你得到 John went to the kitchen, 现在你就可以回答 kitchen 了。

然后就是用这些 gate 来计算最后的 memory cell 和 hidden state。


![](http://upload-images.jianshu.io/upload_images/1667471-82777a63926b56aa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


下面这个是LSTM直观的图表示：

![](http://upload-images.jianshu.io/upload_images/1667471-1a956503f868a17d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

[Theano－LSTM代码.](http://blog.csdn.net/u010223750/article/details/51510897)

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

我是 *不会停的蜗牛* Alice
85后全职主妇
喜欢人工智能，行动派
创造力，思考力，学习力提升修炼进行中
欢迎您的喜欢，关注和评论！
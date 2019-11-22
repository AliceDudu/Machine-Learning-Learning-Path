RNN的高级应用

---
本文结构：

四个问题

1. 每个问题是什么
2. 应用什么模型
3. 模型效果

---

CS224d-Day 11: 
Recursive neural networks -- for different tasks (e.g. sentiment analysis)
[课程链接](https://web.archive.org/web/20160314075834/http://cs224d.stanford.edu/syllabus.html)
[视频链接](https://www.youtube.com/watch?v=24FQOQMcOIY&list=PLlJy-eBtNFt4CSVWYqscHDdP58M3zFHIG&index=11)
[课件链接](https://web.archive.org/web/20160313081419/https://cs224d.stanford.edu/lectures/CS224d-Lecture10.pdf)


---

**四个问题**

这次课主要讲了标准的 Recursive neural networks 模型及其扩展模型在3个问题上的应用和效果，最后的 Tree LSTM 简单地介绍了模型和效果。
这3个问题分别是 Paraphrase detection，Relation classification，Sentiment Analysis。
每个模型都可以应用到任意一个问题上，只不过效果不同，有些模型对一些问题表现会更优一些。

![](http://upload-images.jianshu.io/upload_images/1667471-cf7dae9670ac916f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

### 1.Paraphrase detection

####目的是判断两句话是否具有相同的意思
####用到的模型是标准的RNN

为了解决这个问题，需要思考：

- 怎样比较两个句子？

	通过成对地比较两个句子的短语，这个时候可以用 standard RNN，因为它可以得到一个合理的树结构，也就是句子的短语组成结构。
	
- 怎样用相似度来判断两个句子的意义是一样的？

	如果用两个树结构的顶点去判断，那会丢掉很多中间环节的信息。
	如果只计数两个句子中相似短语的个数，那么会丢掉位置信息，即这些短语出现在什么位置。
	
**所以用 similar matrix 来表示相似度。**

如下图，左边是两个句子，树结构中分别有7个部分和5个部分，右边是由similar matrix到最后结果的过程。

similar matrix 由5行7列组成，颜色深浅表示两个树结构相应部分间的相似度大小。

![](http://upload-images.jianshu.io/upload_images/1667471-3a70eaea34c5656c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


- 上图中，为什么不能直接把 similar matrix 直接投入神经网络中？

	因为这个矩阵的大小是随着输入句子的不同而变化的。

**所以需要引用一个 pooling 层**，它可以将输入的 similar matrix 映射成维度一致的矩阵，然后再投入到 RNN 中。


**最后的效果：**
![](http://upload-images.jianshu.io/upload_images/1667471-251b41afda68b31f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

### 2.Relation Classification


####目的是识别词语之间的关系
尤其是 very ＋ good 这种，very 此时只是一个加强 good 的作用
####用到的模型是标准的 Matrix－Vector RNN

普通的 RNN 中，每个节点都是由向量表示的，在这个 Matrix－Vector RNN 中，每个节点除了向量外自带一个矩阵，在由 left 和 right child 生成 parent 的时候，对彼此作用各自的矩阵后，再去生成 parent。

![](http://upload-images.jianshu.io/upload_images/1667471-e1bae3a4c90b53a2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 用向量和矩阵的区别？

	单独的向量反映不出相互作用这种层次的意义，加上矩阵作用给对方后，可以显示出 very 可以让 good 这种形容词更强的意义。
	
	矩阵是随机初始化的，通过 Back Propagation 和 Forward Propagation 可以不断地学习出来。
	

**最后的效果：**

下图中，横轴是 1-10 星号的电影，纵轴是 not annoying 这样的词出现在相应级别中的比例。

在 not annoying ，not awesome 这两个例子中，绿色的 RNN 没有蓝色的 MV－RNN 表现得好，因为 not annoying 出现在低星级的次数不应该比出现在高星级电影中的次数多。
![](http://upload-images.jianshu.io/upload_images/1667471-e954cdbe462e225a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
	
	
####另外一种问题是因果关系等的判断	


![](http://upload-images.jianshu.io/upload_images/1667471-db6c399d0b443217.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**最后的效果：**

对于这个问题，用了不同的模型和feature来看效果。

在 SVM 用了好多feature，最后的效果是 82.2
POS：part of speech
wordnet 大量人工生成的数据
prefix 等其他形态学的特征
dependency parse feature 不同类型的parser
textrunner 百万的网上数据
Google n－gram 几十亿个 n－gram

单纯用神经网络模型，数据量没那么大的时候，效果不到80％
加入了 POS，WordNet，NER 数据后，变成了 82.4，优于SVM。

数据越多的话，效果越好。

![](http://upload-images.jianshu.io/upload_images/1667471-7b56a8d317069356.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

### 3.Sentiment Analysis


####目的是识别句子表达的情感

####用到的模型是RNTN（Recursive Neural Tensor Network）


- 用 Bag of words 这种方法有缺陷：
	一个 not 后面多个 positive 的词时，应该是否定，结果被判断成肯定。
	前半句否定，后半句肯定，后半句的效果比前半句更强的时候，怎么判断出来。
	
解决方案，一个是更好的数据，一个是更好的模型


**更好的数据：**

人工标注 11,855 个句子的 215,154 个短语，每个短语由不同的人标注 3 次。

下图是标注结果的可视化，横轴是短语的长度，纵轴是各个情感类别的比例。

![](http://upload-images.jianshu.io/upload_images/1667471-422b722ca2ccc1ce.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


**最后的效果：**

可以发现，用新的 tree bank 的模型效果要比原来的好，肯定否定情感分类越准。

![](http://upload-images.jianshu.io/upload_images/1667471-c26fbee343f4006b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



**更好的模型：**
RNTN（Recursive Neural Tensor Network）

这个模型可以让 word 之间有更多的 interaction，‘very good’的词向量的转置和矩阵 V 再和词向量本身作用。

![](http://upload-images.jianshu.io/upload_images/1667471-82172d32533d8fa7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![](http://upload-images.jianshu.io/upload_images/1667471-e896f46058c22dab.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



**最后的效果：**

RNTN 作用在新的 Tree Bank 上效果可以高达 85.4.

![](http://upload-images.jianshu.io/upload_images/1667471-2af9ca75c721949e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

### 4.Semantic Similarity


####目的是识别语义相似性

####用到的模型是 Tree LSTMs

Tree LSTMs 和普通的 LSTMs 的不同之处在于 Tree LSTMs 是从 tree 的结构中进行LSTMs 的建模。

parent 的 hidden层是其 children 的 hidden 层的和，每一个 forget unit 是根据具体的某个节点来计算的，计算最终 cell 时要把所有 forget units 和对应的 cells 相乘并求和，其他部分和普通LSTMs计算方法一样。

![](http://upload-images.jianshu.io/upload_images/1667471-d3dec94879521292.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



**最后的效果：**

![](http://upload-images.jianshu.io/upload_images/1667471-5e0dba295882cc3a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

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
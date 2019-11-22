用 Recursive Neural Networks 得到分析树

---

CS224d-Day 10:
 Recursive neural networks -- for parsing
[课程链接](https://web.archive.org/web/20160314075834/http://cs224d.stanford.edu/syllabus.html)
[视频链接](https://www.youtube.com/watch?v=D4j_9Jn-E8g&index=10&list=PLlJy-eBtNFt4CSVWYqscHDdP58M3zFHIG)
[课件链接](https://web.archive.org/web/20160313082614/https://cs224d.stanford.edu/lectures/CS224d-Lecture9.pdf)

---

本文结构：

- **Recursive NN 是什么**
- **Recursive Neural Networks 和 Recurrent Neural Networks**
- **Recursive NN 可以用来做什么**
- **怎样做到的**
- **算法代码**

---

- **Recursive NN 是什么**

Recursive  Neural Networks 可以用来表达长句子，将一个句子映射到向量空间。

通过分析出句子的 parsing tree 的结构，把一个句子拆分成几个小组成单元，然后可以替换其中的一些部分，进而得到一些相似的句子，比如把这个 NP 名词短语，换成另一个 NP 名词短语。


![](http://upload-images.jianshu.io/upload_images/1667471-e437a2f3b13d0d4d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这些句子由不同的短语组成，但是表达的意思却是一样的，在向量空间中，它们的距离也会很近。例如 ‘the country of my birth’ 和 ‘the place where I was born’ 意思一样，向量空间上的表达也就很近。

![](http://upload-images.jianshu.io/upload_images/1667471-ec38b090a59a333a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


- **Recursive Neural Networks 和 Recurrent Neural Networks**

想要分析数据的 hiearchical structure 的时候，Recursive NN 要比 Recurrent NN 更有效一些。

Recurrent NN 是 Recursive NN 的一种特殊形式，一个 链 可以写成一棵 树 的形式。


![](http://upload-images.jianshu.io/upload_images/1667471-0751851caf61250b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Recursive NN 可以返回树上每个节点的向量表达，Recurrent NN 在任何时候返回一句话的向量。

- **Recursive NN 可以用来做什么**

Recursive NN 可以用一个很好的方式来描述句子。
可以识别句子中的成分，可以通过替换组件来形成同样合理的句子，可以处理歧义问题，分析句子的语法结构，语义结构，理解一段话的指代词的含义。
可以学习到一个句子里哪个组成成分更重要，比如VP比NP更重要。可以学习到哪几个句子意思相近。


1. 当我们需要学习句子的结构的时候，会用 Recursive Neural Networks 来的到 parsing tree。
2. 也可以用来做 sentiment analysis，因为这个情感喜好的结果，不仅仅和单词本身有关，还和句子组成和顺序有关。
3. 还可以用来分析图片的组成，比如它可以分析出房顶，二层楼，一层楼，并组成一个房子。


![](http://upload-images.jianshu.io/upload_images/1667471-fa63d9f1601613f2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



- **怎样做到的**
 
RNN 的输入是句子，输出是一个 parse 树结构。

下图是一个最标准的神经层，W 在整个网络中是一样的。


![](http://upload-images.jianshu.io/upload_images/1667471-806b600a81cf9151.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


有个拓展模型 Syntactically-United RNN，是根据不同的组成成分使用不同的 W


![](http://upload-images.jianshu.io/upload_images/1667471-4091bd1e1d5f95f7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


RNN由三部分组成，根，左叶子，右叶子，也就是一个 Binary Tree。
它的每个节点上由一些神经元组成，神经元的个数由句子的复杂程度决定。
叶子是接收数据的，也就是向量，根是分类和评分的。


![](http://upload-images.jianshu.io/upload_images/1667471-6f6091623029f9ca.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

第一步，会先把句子的结构学习出来。
单词两两组合，进行评分，再作为一个整体，和后面的一个单词组合，再评分。
两个单词如果应该放在一起，就会得到高分，否则分数较低。


![](http://upload-images.jianshu.io/upload_images/1667471-bfe894650a9bf65e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![](http://upload-images.jianshu.io/upload_images/1667471-1151d16ec639d152.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


一个句子会得到多个结构，用 Greedy 选择其中分数最高的，作为最终的树结构。

用 Max Margin 来学习最优的树结构。每个 i 代表一个句子，A(x_i) 是包含 x_i 的所有可能的树，当 y 与 y_i 一样时，delta＝0.


![](http://upload-images.jianshu.io/upload_images/1667471-8241fcc27554d58d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![](http://upload-images.jianshu.io/upload_images/1667471-41f3094690d06da3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


模型训练时，通过比较 labeled 数据，比较合适的结构和预测的结构，用 BTS 使误差达到最小。

![](http://upload-images.jianshu.io/upload_images/1667471-ff1659a4d4b88fea.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


第二步，再为结构中的每个小部分找到合适的语法标签，判断是什么成分的短语，是NP名词短语，VP动词短语，还是PP介词短语等。


- **算法代码**

定义线性的神经元，做内积 W(left + right) + b
用 softmax 对每个点做 classify
node.probs -= np.max(node.probs) 这个技巧可以保证stable

![](http://upload-images.jianshu.io/upload_images/1667471-6fe2001d9453b3cc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下面的代码就是计算红色框里的式子

![](http://upload-images.jianshu.io/upload_images/1667471-eb778ff8d2a84a79.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

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
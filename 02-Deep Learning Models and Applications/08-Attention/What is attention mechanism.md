attention 机制入门

在下面这两篇文章中都有提到 attention 机制：
[使聊天机器人的对话更有营养](http://www.jianshu.com/p/11d7c7772d4c)
[如何自动生成文章摘要](http://www.jianshu.com/p/abc7e13abc21)

今天来看看 attention 是什么。﻿

下面这篇论文算是在NLP中第一个使用attention机制的工作。他们把attention机制用到了神经网络机器翻译（NMT）上，NMT其实就是一个典型的sequence to sequence模型，也就是一个encoder to decoder模型﻿
https://arxiv.org/pdf/1409.0473.pdf﻿

encoder 里面用的是 Bi RNN，这样每个单词的表达不仅能包含前一个单词的信息，还可以包含后一个; 前向RNN按输入序列的顺序，生成同样顺序的隐藏层状态，反向RNN则逆向生成隐藏层状态序列，然后我们将每个时刻的这两个状态合并为一个状态，这样它就既包含当前单词的前一个单词信息，也包含后一个信息; 这个状态之后将被用于 decoder 部分。﻿

![](http://upload-images.jianshu.io/upload_images/1667471-e301bdd418695bac.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这里的条件概率是这样的，﻿

![](http://upload-images.jianshu.io/upload_images/1667471-a87b637551bec8c5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

和一般的encoder decoder区别就是这个条件概率考虑了每个单词的语境向量 c﻿

c 就是由前面得到的 h 计算﻿

![](http://upload-images.jianshu.io/upload_images/1667471-e4bc3242d435cb05.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

权重 alpha 由 e 计算，alpha i j 相当于 y i 是由 x j 翻译而成的概率，这个概率就反映了 hj 的重要性﻿

这里就应用了 attention 机制，这样 decoder 就决定了输入句子中的什么部分需要加以注意﻿

有了注意力机制就不用把所有的输入信息都转化到一个固定长度的向量中﻿

e 是个 score，用来评价 j 时刻的输入和 i 时刻的输出之间的匹配程度，﻿

a 是一个 alignment midel，是一个前向神经网络。

这篇文章中有 seq2seq＋attention 的实现：
[seq2seq 的 keras 实现](http://www.jianshu.com/p/c294e4cb4070)

---
推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的：
[入门问题][TensorFlow][深度学习][强化学习][神经网络][机器学习][自然语言处理][聊天机器人]

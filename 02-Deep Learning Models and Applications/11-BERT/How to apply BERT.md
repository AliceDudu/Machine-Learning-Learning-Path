上一篇文章介绍了 Google 最新的**BERT (Bidirectional Encoder Representations from Transformers) ，这个模型在 11 个 NLP 任务上刷新了纪录。**

Google 还开源了 BERT 的代码：https://github.com/google-research/bert﻿

大家可以下载在维基百科语料上使用 TPU 预训练好的模型，包括中文 BERT 预训练模型。

BERT 模型的训练分为**预训练（Pre-training）和微调（Pre-training）**两步。

**预训练**过程耗时又耗钱，Google 对 BERT 的预训练一般需要 4 到 16 块 TPU 和一周的时间才可以完成。﻿幸好多数情况下我们可以使用 Google 发布的预训练模型，不需要重复构造，

**微调**时可以根据不同的任务，对模型进行相应的扩展，例如对句子进行情感分类时，只需要在 BERT 的输出层的句向量上面加入几个 Dense 层。所以可以固定 BERT 的参数，将它的输出向量当做一个特征用于具体任务。﻿

---

**那么要如何应用 BERT 呢？**

这里介绍一下 bert-as-service ，项目地址：https://github.com/hanxiao/bert-as-service

这个项目将预训练好的 BERT 模型作为一个服务独立运行，**很简单地用几行代码就可以调用服务获取句子、词级别上的向量**，然后将这些向量当做特征信息输入到下游模型。在做具体 NLP 任务时，不需要将整个 BERT 加载到 tf.graph 中，或者可以直接在 scikit-learn, PyTorch, Numpy 中使用 BERT。﻿

这个项目的作者是**肖涵博士，他的 Fashion-MNIST 数据集**大家应该比较熟悉，现在已成为机器学习基准集，在 Github 上超过 4.4K 星。

---

**使用方法﻿很简单：**

1. 下载 Google 的预训练 BERT 模型﻿，可以选择 BERT-Base, Chinese 等任意模型：

https://github.com/google-research/bert#pre-trained-models﻿

解压到某个路径下，例如：` /tmp/english_L-12_H-768_A-12/ ﻿`

2. 开启 BERT 服务﻿

`python app.py -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4﻿`

这个代码将开启一个 4 进程的 BERT 服务，即最高处理来自 4 个客户端的并发请求。如果在某时刻多余 4 个的并发请求，将被暂时放到一个负载均衡中等待执行。﻿

3. 使用客户端获取句子向量编码﻿

唯一需要的文件就是 `service/client.py` ，从中导入 BertClient 。﻿

```
from service.client import BertClient﻿
bc = BertClient()﻿
bc.encode(['First do it', 'then do it right', 'then do it better'])﻿
```

然后就可以得到一个 3 x 768 的 ndarray 结构，每一行代表了一句话的向量编码。也可以通过设置，返回 Python 类型的 List[List[float]] 。﻿

---

推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的：
[入门问题][TensorFlow][深度学习][强化学习][神经网络][机器学习][自然语言处理][聊天机器人]

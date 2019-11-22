什么是 Dropout

为了应对神经网络很容易过拟合的问题，2014年 Hinton 提出了一个神器，
**Dropout: A Simple Way to Prevent Neural Networks from Overfitting **
(original paper: http://jmlr.org/papers/v15/srivastava14a.html)

实验结果：
![](http://upload-images.jianshu.io/upload_images/1667471-90f6e10fc0e6fc0f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**dropout 是指在深度学习网络的训练过程中，按照一定的概率将一部分神经网络单元暂时从网络中丢弃，相当于从原始的网络中找到一个更瘦的网络**

![](http://upload-images.jianshu.io/upload_images/1667471-8e10d8a8e14a2ef4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在大规模的神经网络中有这样两个缺点：1. 费时；2. 容易过拟合

对于一个有 N 个节点的神经网络，有了 dropou t后，就可以看做是 2^N 个模型的集合了，但此时要训练的参数数目却是不变的，这就缓解了费时的问题。

论文中做了这样的类比，无性繁殖可以保留大段的优秀基因，而有性繁殖则将基因随机拆了又拆，破坏了大段基因的联合适应性，但是自然选择中选择了有性繁殖，物竞天择，适者生存，可见有性繁殖的强大。

dropout 也能达到同样的效果，它强迫一个神经单元，和随机挑选出来的其他神经单元共同工作，消除减弱了神经元节点间的联合适应性，增强了泛化能力。

**每层 Dropout 网络和传统网络计算的不同之处：**
![](http://upload-images.jianshu.io/upload_images/1667471-e3bd5356b99dd84a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
**相应的公式：**
![](http://upload-images.jianshu.io/upload_images/1667471-fa9e242426be214f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
**对于单个神经元是这样的：**
![](http://upload-images.jianshu.io/upload_images/1667471-f1aed1f708d50613.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在训练时，每个神经单元都可能以概率 p 去除；
在测试阶段，每个神经单元都是存在的，权重参数w要乘以p，成为：pw。

**看一下在 Keras 里面怎么用 dropout**

问题：binary 分类，根据数据集，识别 rocks 和 mock-mines
数据集下载：存在 sonar.csv 里面，http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data

Baseline 中，NN 具有两个 隐藏层，分别有 60 和 30 个神经元，用 SGD 训练，并用 10-fold cross validation 得到 classification accuracy 为： 86.04%

![](http://upload-images.jianshu.io/upload_images/1667471-2ad8c60c1893ebd5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](http://upload-images.jianshu.io/upload_images/1667471-ff030909810cf1b8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](http://upload-images.jianshu.io/upload_images/1667471-110ce620c6df7263.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在 input 和 第一个隐藏层之间，插入一层 dropout ，rate＝20%，意思是，5个神经元里面有一个被随机去掉后，accuracy 为：82.18%，下降了一点

![](http://upload-images.jianshu.io/upload_images/1667471-e5e4d96420a327c5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


在两个隐藏层之间，第二个隐藏层和 output 层之间加入 dropout 后，accuracy 为：84.00%

![](http://upload-images.jianshu.io/upload_images/1667471-ddbce1af2c8accb8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可见本例并不适合用 dropout 的。


参考资料：
http://blog.csdn.net/stdcoutzyx/article/details/49022443
http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/

---
推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的

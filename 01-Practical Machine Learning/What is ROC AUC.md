什么是 ROC AUC

**本文结构：**

1. 什么是 ROC？
2. 怎么解读 ROC 曲线？
3. 如何画 ROC 曲线？
4. 代码？
5. 什么是 AUC？
6. 代码？


---

ROC 曲线和 AUC 常被用来评价一个二值分类器的优劣。

先来看一下混淆矩阵中的各个元素，在后面会用到：

![](http://upload-images.jianshu.io/upload_images/1667471-50d642acd5cd778a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

**1. ROC ：**

纵轴为 TPR 真正例率，预测为正且实际为正的样本占所有正例样本的比例。
横轴为 FPR 假正例率，预测为正但实际为负的样本占所有负例样本的比例。

![](http://upload-images.jianshu.io/upload_images/1667471-405e26dc57dd7bca.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

对角线对应的是 “随机猜想”
![](http://upload-images.jianshu.io/upload_images/1667471-26a6960b8cfe543b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

当一个学习器的 ROC 曲线被另一个学习器的包住，那么后者性能优于前者。
有交叉时，需要用 AUC 进行比较。


**2. 先看图中的四个点和对角线：**

- 第一个点，(0,1)，即 FPR=0, TPR=1，这意味着 FN（false negative）=0，并且FP（false positive）=0。这意味着分类器很完美，因为它将所有的样本都正确分类。
- 第二个点，(1,0)，即 FPR=1，TPR=0，这个分类器是最糟糕的，因为它成功避开了所有的正确答案。
- 第三个点，(0,0)，即 FPR=TPR=0，即 FP（false positive）=TP（true positive）=0，此时分类器将所有的样本都预测为负样本（negative）。
- 第四个点（1,1），分类器将所有的样本都预测为正样本。
- 对角线上的点表示分类器将一半的样本猜测为正样本，另外一半的样本猜测为负样本。

因此，ROC 曲线越接近左上角，分类器的性能越好。

**3. 如何画 ROC 曲线**

例如有如下 20 个样本数据，Class 为真实分类，Score 为分类器预测此样本为正例的概率。
![](http://upload-images.jianshu.io/upload_images/1667471-8becd4bacaae809c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 按 Score 从大到小排列
- 依次将每个 Score 设定为阈值，然后这 20 个样本的标签会变化，当它的 score 大于或等于当前阈值时，则为正样本，否则为负样本。
- 这样对每个阈值，可以计算一组 FPR 和 TPR，此例一共可以得到 20 组。
- 当阈值设置为 1 和 0 时， 可以得到 ROC 曲线上的 (0,0) 和 (1,1) 两个点。

![](http://upload-images.jianshu.io/upload_images/1667471-67453c7140e06e7c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**4. 代码：**

输入 y 的真实标签，还有 score，设定标签为 2 时是正例：
```
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
```

就会得到相应的  TPR, FPR, 截断点 ：
```
fpr = array([ 0. ,  0.5,  0.5,  1. ])
tpr = array([ 0.5,  0.5,  1. ,  1. ])
thresholds = array([ 0.8 ,  0.4 ,  0.35,  0.1 ])#截断点
```

---

**5. AUC：**

是 ROC 曲线下的面积，它是一个数值，当仅仅看 ROC 曲线分辨不出哪个分类器的效果更好时，用这个数值来判断。
![](http://upload-images.jianshu.io/upload_images/1667471-e521a84a9900b83f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

The AUC value is equivalent to the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example.

从上面定义可知，意思是随机挑选一个正样本和一个负样本，当前分类算法得到的 Score 将这个正样本排在负样本前面的概率就是 AUC 值。AUC 值是一个概率值，AUC 值越大，分类算法越好。

**6. 代码：**

```
import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
roc_auc_score(y_true, y_scores)

0.75
```

---

学习资料：
《机器学习》，周志华
http://alexkong.net/2013/06/introduction-to-auc-and-roc/
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html

---
推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的：
[入门问题][TensorFlow][深度学习][强化学习][神经网络][机器学习][自然语言处理][聊天机器人]

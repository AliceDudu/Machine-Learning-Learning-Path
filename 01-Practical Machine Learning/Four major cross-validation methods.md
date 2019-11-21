四种交叉验证方法

本文结构：

- 什么是交叉验证法？
- 为什么用交叉验证法？
- 主要有哪些方法？优缺点？
- 各方法应用举例？

---

#### **什么是交叉验证法？** 

它的基本思想就是将原始数据（dataset）进行分组，一部分做为训练集来训练模型，另一部分做为测试集来评价模型。

---

#### **为什么用交叉验证法？** 

1. 交叉验证用于评估模型的预测性能，尤其是训练好的模型在新数据上的表，可以在一定程度上减小过拟合。
2. 还可以从有限的数据中获取尽可能多的有效信息。
3. 可以选择出合适的模型

---

#### **主要有哪些方法？** 

- Holdout Method
- K-Fold CV
- Leave One out CV
- Bootstrap Methods

---
 
#### **1. 留出法 （holdout cross validation）**

![](https://upload-images.jianshu.io/upload_images/1667471-aa22db881e9f66f6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这种方法是最简单的交叉验证：

在机器学习任务中，拿到数据后，我们首先会将原始数据集分为三部分：**训练集、验证集和测试集**。
训练集用于训练模型，验证集用于模型的参数选择配置，测试集对于模型来说是未知数据，用于评估模型的泛化能力。

![](http://upload-images.jianshu.io/upload_images/1667471-9db53006d07c7d20.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这个方法操作简单，只需随机把原始数据分为三组即可。

不过如果只做一次分割，它对训练集、验证集和测试集的样本数**比例**，还有分割后数据的分布是否和原始数据集的**分布**相同等因素比较敏感，
**不同的划分会得到不同的最优模型，**
而且分成三个集合后，用于训练的数据**更少**了。

---

 #### **2. k 折交叉验证（k-fold cross validation）** 

于是有了 **k 折交叉验证（k-fold cross validation）** 加以改进：

![](http://upload-images.jianshu.io/upload_images/1667471-7ddeb02e0be14b79.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

k 折交叉验证通过对 k 个不同分组训练的结果进行平均来减少方差，
因此**模型的性能对数据的划分就不那么敏感**。

- 第一步，不重复抽样将原始数据随机分为 k 份。
- 第二步，每一次挑选其中 1 份作为测试集，剩余 k-1 份作为训练集用于模型训练。
- 第三步，重复第二步 k 次，这样每个子集都有一次机会作为测试集，其余机会作为训练集。
- 在每个训练集上训练后得到一个模型，
- 用这个模型在相应的测试集上测试，计算并保存模型的评估指标，
- 第四步，计算 k 组测试结果的平均值作为模型精度的估计，并作为当前 k 折交叉验证下模型的性能指标。

**k 一般取 10，**
数据量小的时候，k 可以设大一点，这样训练集占整体比例就比较大，不过同时训练的模型个数也增多。
数据量大的时候，k 可以设小一点。

---

#### 3. 留一法（Leave one out cross validation）

![](https://upload-images.jianshu.io/upload_images/1667471-acc6b43c023cb955.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


当 k＝m 即样本总数时，叫做 ** 留一法（Leave one out cross validation）**，
每次的测试集都只有一个样本，要进行 m 次训练和预测。

这个方法用于训练的数据只比整体数据集少了一个样本，因此最接近原始样本的分布。
但是训练复杂度增加了，因为模型的数量与原始数据样本数量相同。
一般在数据缺乏时使用。
样本数很多的话，这种方法开销很大。

此外：

1. 多次 k 折交叉验证再求均值，例如：10 次 10 折交叉验证，以求更精确一点。
2. 划分时有多种方法，例如对非平衡数据可以用分层采样，就是在每一份子集中都保持和原始数据集相同的类别比例。
3. 模型训练过程的所有步骤，包括模型选择，特征选择等都是在单个折叠 fold 中独立执行的。

---

#### 4. Bootstrap

![](https://upload-images.jianshu.io/upload_images/1667471-9f51958d66411b23.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

还有一种比较特殊的交叉验证方式，**Bootstrapping：** 通过自助采样法，
即在含有 m 个样本的数据集中，进行 m 次有放回地随机抽样，组成的新数据集作为训练集。

这种方法，有的样本会被多次采样，也会有一次都没有被选择过的样本，原数据集中大概有 36.8% 的样本不会出现在新组数据集中，这些没有被选择过的数据作为验证集。

优点是训练集的样本总数和原数据集一样都是 m，并且仍有约 1/3 的数据不被训练而可以作为测试集，对于样本数少的数据集，就不用再由于拆分得更小而影响模型的效果。
缺点是这样产生的训练集的数据分布和原数据集的不一样了，会引入估计偏差。
此种方法不是很常用，除非数据量真的很少。

##### 36.8% 是怎么得到的？



---

####**各方法应用举例？** 

**1. 留出法 （holdout cross validation）**

下面例子，一共有 150 条数据：

```
>>> import numpy as np
>>> from sklearn.model_selection import train_test_split
>>> from sklearn import datasets
>>> from sklearn import svm

>>> iris = datasets.load_iris()
>>> iris.data.shape, iris.target.shape
((150, 4), (150,))
```

用 train_test_split 来随机划分数据集，其中 40% 用于测试集，有 60 条数据，60% 为训练集，有 90 条数据：

```
>>> X_train, X_test, y_train, y_test = train_test_split(
...     iris.data, iris.target, test_size=0.4, random_state=0)

>>> X_train.shape, y_train.shape
((90, 4), (90,))
>>> X_test.shape, y_test.shape
((60, 4), (60,))
```

用 train 来训练，用 test 来评价模型的分数。

```
>>> clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
>>> clf.score(X_test, y_test)                           
0.96...
```

**2. k 折交叉验证（k-fold cross validation）**

最简单的方法是直接调用 cross_val_score，这里用了 5 折交叉验证：
 
```
>>> from sklearn.model_selection import cross_val_score
>>> clf = svm.SVC(kernel='linear', C=1)
>>> scores = cross_val_score(clf, iris.data, iris.target, cv=5)
>>> scores                                              
array([ 0.96...,  1.  ...,  0.96...,  0.96...,  1.        ])
```
得到最后平均分为 0.98，以及它的 95% 置信区间：

```
>>> print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
Accuracy: 0.98 (+/- 0.03)
```

---

我们可以直接看一下 **K-fold** 是怎样划分数据的：
X 有四个数据，把它分成 2 折，
结果中最后一个集合是测试集，前面的是训练集，
每一行为 1 折：

```
>>> import numpy as np
>>> from sklearn.model_selection import KFold

>>> X = ["a", "b", "c", "d"]
>>> kf = KFold(n_splits=2)
>>> for train, test in kf.split(X):
...     print("%s %s" % (train, test))
[2 3] [0 1]
[0 1] [2 3]
```

同样的数据 X，我们看 **LeaveOneOut** 后是什么样子，
那就是把它分成 4 折，
结果中最后一个集合是测试集，只有一个元素，前面的是训练集，
每一行为 1 折：

```
>>> from sklearn.model_selection import LeaveOneOut

>>> X = [1, 2, 3, 4]
>>> loo = LeaveOneOut()
>>> for train, test in loo.split(X):
...     print("%s %s" % (train, test))
[1 2 3] [0]
[0 2 3] [1]
[0 1 3] [2]
[0 1 2] [3]
```

---

资料：
[https://www.youtube.com/watch?v=vQBIi3Bvt2M&t=105s](https://www.youtube.com/watch?v=vQBIi3Bvt2M&t=105s)
http://scikit-learn.org/stable/modules/cross_validation.html
https://ljalphabeta.gitbooks.io/python-/content/kfold.html
http://www.csuldw.com/2015/07/28/2015-07-28%20crossvalidation/
[https://ww3.arb.ca.gov/research/weekendeffect/carb041300/sld012.htm](https://ww3.arb.ca.gov/research/weekendeffect/carb041300/sld012.htm)


---
推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的
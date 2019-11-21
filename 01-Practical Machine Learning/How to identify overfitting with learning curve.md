用学习曲线 learning curve 来判别过拟合问题

本文结构：

- 学习曲线是什么？
- 怎么解读？
- 怎么画？

---

#### **学习曲线是什么？**
学习曲线就是通过画出不同**训练集大小**时训练集和交叉验证的**准确率**，可以看到模型在新数据上的表现，进而来判断模型是否方差偏高或偏差过高，以及增大训练集是否可以减小过拟合。

---

#### **怎么解读？**

![](http://upload-images.jianshu.io/upload_images/1667471-cc0db48e0b91b13f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

当训练集和测试集的**误差收敛但却很高**时，为高偏差。
左上角的偏差很高，训练集和验证集的准确率都很低，很可能是欠拟合。
我们可以增加模型参数，比如，构建更多的特征，减小正则项。
此时通过增加数据量是不起作用的。

当训练集和测试集的**误差之间有大的差距**时，为高方差。
当训练集的准确率比其他独立数据集上的测试结果的准确率要高时，一般都是过拟合。
右上角方差很高，训练集和验证集的准确率相差太多，应该是过拟合。
我们可以增大训练集，降低模型复杂度，增大正则项，或者通过特征选择减少特征数。

理想情况是是找到偏差和方差都很小的情况，即收敛且误差较小。

---

#### **怎么画？**

在画学习曲线时，**横轴为训练样本的数量，纵轴为准确率。**

![](http://upload-images.jianshu.io/upload_images/1667471-1f5808d0f1e5324a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

例如同样的问题，左图为我们用 **naive Bayes** 分类器时，效果不太好，分数大约收敛在 0.85，此时增加数据对效果没有帮助。

右图为 **SVM（RBF kernel**），训练集的准确率很高，验证集的也随着数据量增加而增加，不过因为训练集的还是高于验证集的，有点过拟合，所以还是需要增加数据量，这时增加数据会对效果有帮助。

---

#### 上图的代码如下：

模型这里用 `GaussianNB` 和 `SVC` 做比较，
模型选择方法中需要用到 `learning_curve` 和交叉验证方法 `ShuffleSplit`。

```
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
```

首先定义画出学习曲线的方法，
核心就是调用了  `sklearn.model_selection` 的 `learning_curve`，
学习曲线返回的是 `train_sizes, train_scores, test_scores`，
画训练集的曲线时，横轴为 `train_sizes`, 纵轴为 `train_scores_mean`，
画测试集的曲线时，横轴为 `train_sizes`, 纵轴为 `test_scores_mean`：

```
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    ~~~
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)  
    ~~~     
```

在调用 `plot_learning_curve` 时，首先定义交叉验证 cv 和学习模型 estimator。

这里交叉验证用的是 `ShuffleSplit`， 它首先将样例打散，并随机取 20％ 的数据作为测试集，这样取出 100 次，最后返回的是 `train_index, test_index`，就知道哪些数据是 train，哪些数据是 test。

estimator 用的是 GaussianNB，对应左图：

```
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = GaussianNB()
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
```

再看 estimator 是 SVC 的时候，对应右图：

```
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)
```



---

资料：
http://scikit-learn.org/stable/modules/learning_curve.html#learning-curve
http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

---
推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的
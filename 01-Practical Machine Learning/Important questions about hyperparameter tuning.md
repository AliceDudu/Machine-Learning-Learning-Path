机器学习面试题集 - 超参数调优

![](http://upload-images.jianshu.io/upload_images/1667471-cdc3a6e4294b7b82.jpg)


#### 超参数搜索算法一般包括哪几个要素

  目标函数

  搜索范围

  算法的其他参数



-------

### 超参数有哪些调优方法？

#### 网格搜索

    给出一个搜索范围后，遍历所有点，找出最优值

    缺点：耗时

    对策：将搜索范围和步长先设置的大一些，锁定最优值的范围。

        再逐渐缩小范围和步长，更精确的确定最优值

    缺点：可能会错过全局最优值

#### 随机搜索

    给定一个搜索范围后，从中随机的选择样本点。

    缺点：可能会错过全局最优值

#### 贝叶斯优化算法

    通过学习目标函数的形状，找到影响最优值的参数。

     算法：首先根据先验分布，假设一个搜集函数。再用每个新的样本点，更新目标函数的先验分布。由后验分布得到全局最值可能的位置

     缺点：容易陷入局部最优值，因为找到了一个局部最优值，会在该区域不断采样

     对策：在还未取样的区域进行探索，在最可能出现全局最值的区域进行采样

---

下面来具体看看如何用 **网格搜索**(grid search) 对 SVM 进行调参。

网格搜索实际上就是暴力搜索：
首先为想要调参的参数设定一组候选值，然后网格搜索会穷举各种参数组合，根据设定的评分机制找到最好的那一组设置。

---

以支持向量机分类器 SVC 为例，用 GridSearchCV 进行调参：

```
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
```

**1. 导入数据集，分成 train 和 test 集：**

```
digits = datasets.load_digits()

n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

```

**2. 备选的参数搭配有下面两组，并分别设定一定的候选值：**
例如我们用下面两个 grids：
kernel＝'rbf', gamma, 'C'
kernel＝'linear', 'C'

```
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
```

**3. 定义评分方法为：**

```
scores = ['precision', 'recall']
```

**4. 调用 GridSearchCV**，

将 `SVC(), tuned_parameters, cv=5`, 还有 scoring 传递进去，
用训练集训练这个学习器 clf，
再调用 `clf.best_params_` 就能直接得到**最好的参数**搭配结果，

例如，在 precision 下，
返回最好的参数设置是：`{'C': 10, 'gamma': 0.001, 'kernel': 'rbf'}`

还可以通过 `clf.cv_results_` 的 'params'，'mean_test_score'，看一下具体的参数间不同数值的组合后**得到的分数**是多少：
结果中可以看到最佳的组合的分数为：0.988 (+/-0.017)

![](http://upload-images.jianshu.io/upload_images/1667471-79d0278a287c3d5d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

还可以通过 `classification_report` 打印在**测试集上的预测结果** `clf.predict(X_test)` 与真实值 `y_test` 的分数：

![](http://upload-images.jianshu.io/upload_images/1667471-3fa04971f2606d72.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


```
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

	 # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    # 用训练集训练这个学习器 clf
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    
    # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
    print(clf.best_params_)
    
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    
    # 看一下具体的参数间不同数值的组合后得到的分数是多少
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
              
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    
    # 打印在测试集上的预测结果与真实值的分数
    print(classification_report(y_true, y_pred))
    
    print()
```

---

相关文章：

[用验证曲线 validation curve 选择超参数](http://www.jianshu.com/p/6d4b7f3b7c14)
[为什么要用交叉验证](http://www.jianshu.com/p/40541aa440c7)
[用学习曲线 learning curve 来判别过拟合问题](http://www.jianshu.com/p/d89dee94e247)


---

大家好！我是 Alice，欢迎进入 机器学习面试题集 系列！

这个系列会以《百面机器学习》的学习笔记为主线，除了用导图的形式提炼出精华，还会对涉及到的重要概念进行更深度的解释，顺便也梳理一下机器学习的知识体系。

**欢迎关注我，一起交流学习！**


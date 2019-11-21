
用 Grid Search 对 SVM 进行调参

上一次用了**验证曲线**来找最优超参数。

[用验证曲线 validation curve 选择超参数](http://www.jianshu.com/p/6d4b7f3b7c14)

今天来看看**网格搜索**(grid search)，也是一种常用的找最优超参数的算法。

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

相关阅读：

[为什么要用交叉验证](http://www.jianshu.com/p/40541aa440c7)
[用学习曲线 learning curve 来判别过拟合问题](http://www.jianshu.com/p/d89dee94e247)
[用验证曲线 validation curve 选择超参数](http://www.jianshu.com/p/6d4b7f3b7c14)

---
推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的：
[入门问题][TensorFlow][深度学习][强化学习][神经网络][机器学习][自然语言处理][聊天机器人]
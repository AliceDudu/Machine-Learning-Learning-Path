机器学习中常用评估指标汇总

评估指标 Evaluation metrics 可以说明模型的性能，辨别模型的结果。

我们建立一个模型后，计算指标，从指标获取反馈，再继续改进模型，直到达到理想的准确度。在预测之前检查模型的准确度至关重要，而不应该建立一个模型后，就直接将模型应用到看不见的数据上。

今天先来简单介绍几种回归和分类常用的评估方法。

---

###回归：

#####均方误差：

![](http://upload-images.jianshu.io/upload_images/1667471-e3ba89798e351f3e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](http://upload-images.jianshu.io/upload_images/1667471-f1e5d4a76b640377.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中 D 为数据分布，p 为概率密度函数。

```
from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
mean_squared_error(y_true, y_pred)

0.375
```

---

###分类：

#####二分类 and 多分类：

**错误率**
![](http://upload-images.jianshu.io/upload_images/1667471-6b5029c96df86527.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**精度**
![](http://upload-images.jianshu.io/upload_images/1667471-e89525de0465c241.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

#####二分类

#####混淆矩阵：

![](http://upload-images.jianshu.io/upload_images/1667471-50d642acd5cd778a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
from sklearn.metrics import confusion_matrix
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

[[71  1]
[ 2 40]]
```

单纯用 错误率，精度 是无法知道下面的问题时：

**查准率**：
应用场景－当你想知道“挑出的西瓜中有多少比例是好瓜”
![](http://upload-images.jianshu.io/upload_images/1667471-52d1001684011879.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
from sklearn.metrics import precision_score
from sklearn.metrics  import recall_score, f1_score
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))

Precision: 0.976
```

**查全率：**
应用场景－当你想知道“所有好瓜盅有多少比例被挑出来了”
![](http://upload-images.jianshu.io/upload_images/1667471-72fe0e5e5300f997.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))

Recall: 0.952
```

**P－R 图：**
当一个学习器的 P－R 曲线被另一个学习器的包住，那么后者性能优于前者。
有交叉时，需要在具体的查准率或者查全率下进行比较。
![](http://upload-images.jianshu.io/upload_images/1667471-d1ba7329f04ca8bb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


**平衡点 (Break Event Point BEP)：**
即上图中三个红点。
综合考虑查准率，查全率的度量
当 查准率＝查全率 时的点，谁大谁比较优。

**F1 度量：**
也是综合考虑查准率，查全率的度量，比 BEP 更常用：
![](http://upload-images.jianshu.io/upload_images/1667471-39ca529cd9602cf4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

F1: 0.964
```

**Fβ：**
可以表达对查准率，查全率的不同重视度，
β > 1 则查全率有更大影响，β < 1 则查准率有更大影响，β ＝ 1 则为 F1。
![](http://upload-images.jianshu.io/upload_images/1667471-6e00917010015c10.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

#####One vs. All (OvA) 分类问题
这时会在 n 个二分类问题上综合考虑查准率，查全率。

**宏～ ：先在每个混淆矩阵上计算率，再求平均**

**宏查准率**
![](http://upload-images.jianshu.io/upload_images/1667471-bd1250ed3dd94002.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


**宏查全率**
![](http://upload-images.jianshu.io/upload_images/1667471-12230de215f61a4e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**宏 F1**
![](http://upload-images.jianshu.io/upload_images/1667471-8b17379d376fd0e8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


**微～ ：先将各个混淆矩阵上对应元素求平均，再计算率**

**微查准率**
![](http://upload-images.jianshu.io/upload_images/1667471-fbc90b0680e1c69c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


**微查全率**
![](http://upload-images.jianshu.io/upload_images/1667471-475ba4eba200416f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**微 F1**
![](http://upload-images.jianshu.io/upload_images/1667471-00119f9ecbda744a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

**ROC ：**
反映敏感性和特异性连续变量的综合指标，roc曲线上每个点反映着对同一信号刺激的感受性。

纵轴为 TPR 真正例率，预测为正且实际为正的样本占所有正例样本的比例
横轴为 FPR 假正例率。预测为正但实际为负的样本占所有负例样本的比例

![](http://upload-images.jianshu.io/upload_images/1667471-405e26dc57dd7bca.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

对角线对应的是 “随机猜想”
![](http://upload-images.jianshu.io/upload_images/1667471-26a6960b8cfe543b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

当一个学习器的 ROC 曲线被另一个学习器的包住，那么后者性能优于前者。
有交叉时，需要用 AUC 进行比较。

**AUC：**
ROC 曲线下的面积
![](http://upload-images.jianshu.io/upload_images/1667471-e521a84a9900b83f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
roc_auc_score(y_true, y_scores)

0.75
```

---

#####代价敏感
现实任务中，当不同类型的错误具有不同的影响后果时，它们的代价也是不一样的。

此时，可以设定 
**代价矩阵 cost matrix：**
如果将第 0 类预测为 第 1 类造成的损失更大，则 cost01 > cost10，相反将第 1 类预测为 第 0 类造成的损失更大，则 cost01 < cost10 :
![](http://upload-images.jianshu.io/upload_images/1667471-79a3f891d94824d3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**则带有“代价敏感”的错误率为：**
![](http://upload-images.jianshu.io/upload_images/1667471-2551eaaba5d67b1f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中 0 为正类，1 为反类，D＋ 为正例子集合，D－ 为反例子集合。


**代价曲线 cost curve：**
非均等代价下，反应学习器的期望总体代价。
横轴为取值为［0，1］的正例概率代价：
![](http://upload-images.jianshu.io/upload_images/1667471-adeea975d8612789.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


纵轴为取值为［0，1］的归一化代价：
![](http://upload-images.jianshu.io/upload_images/1667471-cddb1983e5fc0961.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中 p 为正例的概率，FPR ＝ 1 - TPR。

![](http://upload-images.jianshu.io/upload_images/1667471-ab2c93bf1fc8b012.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

资料：
机器学习
Python Machine Learning

---
推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的：
[入门问题][TensorFlow][深度学习][强化学习][神经网络][机器学习][自然语言处理][聊天机器人]
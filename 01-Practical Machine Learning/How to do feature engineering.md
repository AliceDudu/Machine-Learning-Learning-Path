特征工程怎么做

在工业应用中，feature 比算法重要，数据比 feature 重要，有很多 kaggle 参赛者分享经验时也是说 feature engineering 很重要，今天来写一写特征工程相关的。

本文结构

1. Feature Engineering 是什么
2. 有什么用
3. 怎么用
4. 实际应用

---

### 1. 是什么

[参考](https://www.youtube.com/watch?v=CAnEJ42eEYA)

先用例子来直观地了解一下

例如要分析声音，直接拿来数据，是什么都学不到的，需要进行 fourier 变换

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/1667471-6313431ba3fb2d9d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


例如分析图片，判断这个图片是不是苹果，可以选择 形状，颜色分布，边 来作为feature

![Paste_Image.png](http://upload-images.jianshu.io/upload_images/1667471-913bfa5ad5589102.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


特征工程就是选择一些表示典型特征的数据，来替代原始数据作为模型的输入，进而得到比较好的输出效果。


### 2. 有什么用

特征越好，模型的性能越好，而且可以不用很复杂的数学模型也能达到不错的效果

### 3. 怎么做

[参考](http://www.csuldw.com/2015/10/24/2015-10-24%20feature%20engineering/)


特征工程是个过程，包括三个子模块：特征构建->特征提取->特征选择

**特征构建：**根据原始数据构建新的特征，需要找出一些具有物理意义的特征。

**特征提取：**自动地构建新的特征，将原始特征转换为一组具有明显物理意义或者统计意义或核的特征。例如 Gabor、几何特征、纹理等。

常用的方法有：

- PCA (Principal component analysis，主成分分析)
- ICA (Independent component analysis，独立成分分析)
- LDA （Linear Discriminant Analysis，线性判别分析）


**特征选择：**从特征集合中挑选一组最具统计意义的特征子集，把无关的特征删掉，从而达到降维的效果

常用的方法：

- filter（刷选器）方法：Pearson相关系数，Gini-index（基尼指数），IG（信息增益）等

- wrapper（封装器）：有逐步回归（Stepwise regression）、向前选择（Forward selection）和向后选择（Backward selection）等

- Embeded(集成方法)：Regularization，或者使用决策树思想，Random Forest和Gradient boosting等



这篇文章[《使用sklearn做特征工程》](http://www.voidcn.com/blog/xw_classmate/article/p-5956345.html), 使用sklearn中的IRIS（鸢尾花）数据集来对特征处理的过程进行了说明，包括包的使用，数据预处理，还有上面提到的一些特征选择方法的 python 代码和应用例子：

**特征选择部分：**

- Filter：方差选择，相关系数法，卡方检验，互信息法
- Wrapper：递归特征消除法
- Embedded：基于L1，L2惩罚项的特征选择法，基于树模型GBDT的特征选择法

**特征提取部分：**

降维：主成分分析法（PCA），线性判别分析法（LDA）



![](http://upload-images.jianshu.io/upload_images/1667471-c4663562c0c38f2c.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

[图片来源：](http://blog.jasonding.top/2015/07/30/Feature%20Engineering/%E3%80%90%E7%89%B9%E5%BE%81%E5%B7%A5%E7%A8%8B%E3%80%91%E7%89%B9%E5%BE%81%E5%B7%A5%E7%A8%8B%E6%8A%80%E6%9C%AF%E4%B8%8E%E6%96%B9%E6%B3%95/)



对于上面提到的方法，还需要学习一下各自的含义原理和应用场景。

这是 CS 294: Practical Machine Learning的一个课件，讲了 [feature engineer 主要方法的原理和应用结果](https://people.eecs.berkeley.edu/~jordan/courses/294-fall09/lectures/feature/slides.pdf)


此外，scikit learn 有关于 [Feature extraction 的讲解和代码例子](http://scikit-learn.org/stable/modules/feature_extraction.html)，可以学习学习


### 4. 实际应用

伯乐在线上的一篇文章 关于[推荐系统中的特征工程](http://blog.jobbole.com/74951/), 以个性化推荐系统为例，介绍了特征工程在实际的问题里是怎么做的。

下面是Quora上对[‘What are some best practices in Feature Engineering?’](https://www.quora.com/What-are-some-best-practices-in-Feature-Engineering)这个问题的一个回答，讲了一些实际的经验，应用时可以作为一点启发:


1. 理解数据
	1. 特征是连续的还是离散
	2. 特征数据的分布如何
	3. 分布依赖的因素
	4. 是否有数据缺失，重复，交叉
	5. 特征的来源
	6. 数据是实时的吗
2. 头脑风暴更多的特征
	好的特征具有下面特点
	1. 能够直观地解释
	2. 可以被计算
	3. 是很好地观察数据的方式
	例如：用户是否成为网站的注册用户？过去一周花费多少时间在这个网站上？在所有注册用户花费时间的分布中，这个用户处于什么位置？
3. 检验你的猜测
	建模完成后要检验，数据分割，交叉检验等。

4. 围绕你的目标思考
	无论是做回归，分类，还是聚类问题，你的猜想和建模都是为了解决这个问题服务的。

5. 坚持 RFMVT 原则
	- Recency. Signals how old certain event is.
	- Frequency. Signals how often does certain events occur.
	- Monetary. Any numerical representation of direct of indirect business value of an example.
	- Variety. How many distinct items are found for certain type of an example.
	- Tenure. How much time has elapsed since the first appearance of certain example or of an example of certain type.

6. 不要过于工程化
	快速迭代，最有效的方式就是提出正确的问题。


---

我是 *不会停的蜗牛* Alice
85后全职主妇
喜欢人工智能，行动派
创造力，思考力，学习力提升修炼进行中
欢迎您的喜欢，关注和评论！
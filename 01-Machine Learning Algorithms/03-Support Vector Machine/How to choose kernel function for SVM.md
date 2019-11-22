SVM 的核函数选择和调参

---
本文结构：
1. 什么是核函数
2. 都有哪些 & 如何选择
3. 调参
---



###1. 什么是核函数

核函数形式 K(x, y) = <f(x), f(y)>，
其中 x, y  为 n 维，f 为 n 维到 m 维的映射，<f(x), f(y)> 表示内积。

在用SVM处理问题时，如果数据线性不可分，希望通过 将输入空间内线性不可分的数据 映射到 一个高维的特征空间内，使数据在特征空间内是线性可分的，这个映射记作 ϕ(x)，

![](https://upload-images.jianshu.io/upload_images/1667471-5fa8493ba11e17c3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

之后优化问题中就会有内积 ϕi⋅ϕj，
这个内积的计算维度会非常大，因此引入了核函数，
kernel 可以帮我们很快地做一些计算, 否则将需要在高维空间中进行计算。

---

###2. 都有哪些 & 如何选择

![](https://upload-images.jianshu.io/upload_images/1667471-5369977d45099388.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下表列出了 9 种核函数以及它们的用处和公式，常用的为其中的前四个：linear，Polynomial，RBF，Sigmoid

|   核函数      | 用处    |  公式  | 
    | --------   | -----:   | :----: | 
 | linear kernel       |  线性可分时，特征数量多时，样本数量多再补充一些特征时，linear kernel可以是RBF kernel的特殊情况  |  ![](https://upload-images.jianshu.io/upload_images/1667471-55f50ec52f369bc3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | 
 | Polynomial kernel       | image processing，参数比RBF多，取值范围是(0,inf) |  ![](https://upload-images.jianshu.io/upload_images/1667471-9d697255b7ddde54.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | 
| Gaussian radial basis function (RBF)       | 通用，线性不可分时，特征维数少 样本数量正常时，在没有先验知识时用，取值在[0,1] | ![](https://upload-images.jianshu.io/upload_images/1667471-0370c098adda9e32.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | 
 | Sigmoid kernel       | 生成神经网络，在某些参数下和RBF很像，可能在某些参数下是无效的 |  ![](https://upload-images.jianshu.io/upload_images/1667471-b1395552a83502bd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | 
 | Gaussian kernel      | 通用，在没有先验知识时用 | ![](https://upload-images.jianshu.io/upload_images/1667471-bf7c0436c0522324.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | 
 | Laplace RBF kernel      | 通用，在没有先验知识时用 | ![](https://upload-images.jianshu.io/upload_images/1667471-44741fb115bb627c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | 
 | Hyperbolic tangent kernel       | neural networks中用 |  ![](https://upload-images.jianshu.io/upload_images/1667471-fc96e03272076b29.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | 
 | Bessel function of the first kind Kernel      | 可消除函数中的交叉项 | ![](https://upload-images.jianshu.io/upload_images/1667471-b5a690e7ba8ccaa0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  | 
 | ANOVA radial basis kernel       |  回归问题 | ![](https://upload-images.jianshu.io/upload_images/1667471-5951d3392d6359ab.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  | 
 | Linear splines kernel in one-dimension       | text categorization，回归问题，处理大型稀疏向量 | ![](https://upload-images.jianshu.io/upload_images/1667471-3939d6340e77ecee.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  | 

**其中 linear kernel  和 RBF kernel 在线性可分和不可分的对比可视化例子如下：**

|          |  linear kernel     |  RBF kernel  | 
    | --------   | -----:   | :----: | 
 |   线性可分   | ![](https://upload-images.jianshu.io/upload_images/1667471-159acf34f092d7d7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) |  ![](https://upload-images.jianshu.io/upload_images/1667471-f0d1e7e69b9121d3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | 
 | 线性不可分   | ![](https://upload-images.jianshu.io/upload_images/1667471-a0675c4d8ee80dcb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) |  ![](https://upload-images.jianshu.io/upload_images/1667471-bd787ee902c437c2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | 

---

###3. 调参

在 sklearn 中可以用 grid search 找到合适的 kernel，以及它们的 gamma，C 等参数，那么来看看各 kernel 主要调节的参数是哪些：

|   核函数      | 公式    | 调参   | 
    | --------   | -----:   | :----: | 
 | linear kernel       |  ![](https://upload-images.jianshu.io/upload_images/1667471-55f50ec52f369bc3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  |   | 
 | Polynomial kernel       | ![](https://upload-images.jianshu.io/upload_images/1667471-9d697255b7ddde54.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) |  -d：多项式核函数的最高次项次数，-g：gamma参数，-r：核函数中的coef0  | 
| Gaussian radial basis function (RBF)       | ![](https://upload-images.jianshu.io/upload_images/1667471-0370c098adda9e32.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) | -g：gamma参数，默认值是1/k | 
 | Sigmoid kernel       |  ![](https://upload-images.jianshu.io/upload_images/1667471-5289e8077925cb8d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) |  -g：gamma参数，-r：核函数中的coef0 | 

其中有两个重要的参数，即 C（惩罚系数） 和 gamma，
gamma 越大，支持向量越少，gamma 越小，支持向量越多。
而支持向量的个数影响训练和预测的速度。
C 越高，容易过拟合。C 越小，容易欠拟合。

---

学习资料：
https://data-flair.training/blogs/svm-kernel-functions/
https://www.quora.com/What-are-kernels-in-machine-learning-and-SVM-and-why-do-we-need-them
https://www.zhihu.com/question/21883548
https://www.quora.com/How-do-I-select-SVM-kernels

---

推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的：
[入门问题][TensorFlow][深度学习][强化学习][神经网络][机器学习][自然语言处理][聊天机器人]

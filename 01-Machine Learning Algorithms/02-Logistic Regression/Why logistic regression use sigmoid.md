Logistic regression  为什么用 sigmoid ？


假设我们有一个线性分类器：

![](https://upload-images.jianshu.io/upload_images/1667471-729db3fbc23df36f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们要求得合适的 W ，使 0-1 loss 的期望值最小，即下面这个期望最小：

![](https://upload-images.jianshu.io/upload_images/1667471-43d0c3700709eb4d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

一对 x y 的 0-1 loss 为： 

![](https://upload-images.jianshu.io/upload_images/1667471-8741799c86184c41.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在数据集上的 0-1 loss 期望值为：

![](https://upload-images.jianshu.io/upload_images/1667471-1e1d0b549cb14bb3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

由 链式法则 将概率p变换如下：

![](https://upload-images.jianshu.io/upload_images/1667471-96455a368ddd64f8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

为了最小化 R（h），只需要对每个 x 最小化它的 conditional risk：

![](https://upload-images.jianshu.io/upload_images/1667471-8dd48fce8f13c1d1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

由 0-1 loss 的定义，当 h（x）不等于 c 时，loss 为 1，否则为 0，所以上面变为：

![](https://upload-images.jianshu.io/upload_images/1667471-a0179412c13c2b77.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


又因为 

![](https://upload-images.jianshu.io/upload_images/1667471-fe61c0a261a9611e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

所以：


![](https://upload-images.jianshu.io/upload_images/1667471-6f5996b4dfa42b9e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


为了使 条件风险 最小，就需要 p 最大，也就是需要 h 为：

![](https://upload-images.jianshu.io/upload_images/1667471-fc2697745c0dd182.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

上面的问题等价于 找到 c＊，使右面的部分成立：

![](https://upload-images.jianshu.io/upload_images/1667471-7be75744fe0ec560.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

取 log ：

![](https://upload-images.jianshu.io/upload_images/1667471-909fdeec72a3d0e6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在二分类问题中，上面则为：

![](https://upload-images.jianshu.io/upload_images/1667471-c4e693a157a7a536.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

即，我们得到了 log-odds ratio ！

接下来就是对 log-odds ratio 进行建模，最简单的就是想到线性模型：

![](https://upload-images.jianshu.io/upload_images/1667471-96797948f7e5a9d6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

则：

![](https://upload-images.jianshu.io/upload_images/1667471-4c9f37d11dfda97a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

于是得到 sigmoid 函数：

![](https://upload-images.jianshu.io/upload_images/1667471-86a01edfe3cfa968.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

由此可见，log-odds 是个很自然的选择，sigmoid 是对 log-odds 的线性建模。

学习资料：
https://onionesquereality.wordpress.com/2016/05/18/where-does-the-sigmoid-in-logistic-regression-come-from/
https://stats.stackexchange.com/questions/162988/why-sigmoid-function-instead-of-anything-else
https://ask.julyedu.com/question/85100








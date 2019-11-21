图解何为CNN

[参考](https://www.youtube.com/watch?v=JiN9p5vWHDY&list=PLjJh1vlSEYgvZ3ze_4pxKHNh1g5PId36-&index=7)

###  CNN － Convolutional Neural Networks

是近些年在机器视觉领域很火的模型，最先由 Yan Lecun 提出。
如果想学细节可以看 Andrej Karpathy 的 cs231n 。


**How does it work?**

给一张图片，每个圆负责处理图片的一部分。
这些圆就组成了一个 filter。
filter 可以识别图片中是否存在指定的 pattern，以及在哪个区域存在。

![](http://upload-images.jianshu.io/upload_images/1667471-7f95298530911db5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



下图中有4个filter，每个filter的平行的点会负责图片上相同的区域。

![](http://upload-images.jianshu.io/upload_images/1667471-1c254c5952bedd14.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


神经元利用 convolution 的技术查找pattern，简单地理解就是用 filter 的形式去查找图片是否具有某种 pattern。

![](http://upload-images.jianshu.io/upload_images/1667471-ae38003a512ef470.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

weights 和 bias 对模型的效果起着重要的作用。

把白圆圈换成神经元，就是CNN的样子。

![](http://upload-images.jianshu.io/upload_images/1667471-b19968905fc5393a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


Convolution层的神经元之间没有联系，它们各自都只连接inputs。

![](http://upload-images.jianshu.io/upload_images/1667471-b2f246e2b8d8b599.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



同一层的神经元用相同的 weights 和 bias，这样同一层的神经元就可以抓取同样的pattern，只不过是在图片上的不同的区域。

![](http://upload-images.jianshu.io/upload_images/1667471-cb263b0219376ede.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

接下来是 ReLU（Rectified Linear Unit） 层和 Pooling 层，它们用来构建由 convolution 层找到的 pattern。

CNN 也用 Back propagation 训练，所以也有 vanishing gradient 的可能。而 ReLU 作为激活函数的话，gradients会大体保持常值的样子，这样就不会在关键的那几层有很明显的下降。

Pooling 层是用来降维的。
经过 convolution 和 ReLU 的作用后，会有越来越复杂的形式，所以Pooling 层负责提取出最重要的 pattern，进而提高时间空间的效率。

![](http://upload-images.jianshu.io/upload_images/1667471-184e62cdfca7d503.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


这三层可以提取出有用的 pattern，但它们并不知道这些 pattern 是什么。
所以接着是 Fully Connected 层，它可以对数据进行分类。

![](http://upload-images.jianshu.io/upload_images/1667471-dd654a4521a8a7d2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


一个典型的 Deep CNN 由若干组 Convolution－ReLU－Pooling 层组成。
![](http://upload-images.jianshu.io/upload_images/1667471-e9bb3620b4d05835.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


但CNN也有个缺点，因为它是监督式学习，所以需要大量的有标签的数据。


---

我是 *不会停的蜗牛* Alice
85后全职主妇
喜欢人工智能，行动派
创造力，思考力，学习力提升修炼进行中
欢迎您的喜欢，关注和评论！
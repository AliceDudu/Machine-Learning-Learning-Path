为什么要做  batch normalization

#### 首先看一下什么是一般的 normalization：

在机器学习中，我们需要对输入的数据做预处理，
可以用  normalization 归一化 ，或者 standardization 标准化，
用来将数据的不同 feature 转换到同一范围内，
normalization 归一化 ：将数据转换到 [0, 1] 之间，
standardization 标准化：转换后的数据符合标准正态分布

![](https://upload-images.jianshu.io/upload_images/1667471-83d7b9c16b5f3ca1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

#### 为什么需要做归一化 标准化等 feature scaling？

![](https://upload-images.jianshu.io/upload_images/1667471-887d0ba0a5899e6f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

因为如果不做这样的处理，不同的特征具有不同数量级的数据，它们对线性组合后的结果的影响所占比重就很不相同，数量级大的特征显然影响更大。
进一步体现在损失函数上，影响也更大，可以看一下，feature scaling 之前，损失函数的切面图是椭圆的，之后就变成圆，无论优化算法在何处开始，都更容易收敛到最优解，避免了很多弯路。

---

#### 在神经网络中，不仅仅在输入层要做 feature scaling，在隐藏层也需要做。

尤其是在神经网络中，特征经过线性组合后，还要经过激活函数，
如果某个特征数量级过大，在经过激活函数时，就会提前进入它的饱和区间，
即不管如何增大这个数值，它的激活函数值都在 1 附近，不会有太大变化，
这样激活函数就对这个特征不敏感。
在神经网络用 SGD 等算法进行优化时，不同量纲的数据会使网络失衡，很不稳定。

![](https://upload-images.jianshu.io/upload_images/1667471-6a8771ef60976245.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在神经网络中，这个问题不仅发生在输入层，也发生在隐藏层，
因为前一层的输出值，对后面一层来说，就是它的输入，而且也要经过激活函数，

![](https://upload-images.jianshu.io/upload_images/1667471-6d5da1dcb7f66d01.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

#### 所以就需要做 batch normalization

![](https://upload-images.jianshu.io/upload_images/1667471-03ec6a58ffc4e8ce.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

就是在前一层的线性输出 z 上做  normalization：需要求出这一 batch 数据的平均值和标准差，
然后再经过激活函数，进入到下一层。

![](https://upload-images.jianshu.io/upload_images/1667471-762131131527e284.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

#### 在 Keras 可以这样应用：

```
# import BatchNormalization
from keras.layers.normalization import BatchNormalization

# instantiate model
model = Sequential()

# we can think of this chunk as the input layer
model.add(Dense(64, input_dim=14, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.5))

# we can think of this chunk as the hidden layer    
model.add(Dense(64, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.5))

# we can think of this chunk as the output layer
model.add(Dense(2, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

# setting up the optimization of our weights 
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)

# running the fitting
model.fit(X_train, y_train, nb_epoch=20, batch_size=16, show_accuracy=True, validation_split=0.2, verbose = 2)
```

---
 
学习资料：
https://arxiv.org/pdf/1502.03167v3.pdf
https://www.youtube.com/watch?v=BZh1ltr5Rkg
https://www.youtube.com/watch?v=dXB-KQYkzNU
https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras
https://youtu.be/-5hESl-Lj-4

---

推荐阅读历史技术博文链接汇总

http://www.jianshu.com/p/28f02bb59fe5

也许可以找到你想要的：

[入门问题][TensorFlow][深度学习][强化学习][神经网络][机器学习][自然语言处理][聊天机器人]

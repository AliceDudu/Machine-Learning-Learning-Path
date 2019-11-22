## 权重初始化的几个方法


---

### 我们知道，神经网络的训练大体可以分为下面几步：

1. 初始化 weights 和 biases﻿
2. 前向传播，用 input X, weights W ，biases b, 计算每一层的 Z 和 A，最后一层用 sigmoid, softmax 或 linear function 等作用 A 得到预测值﻿ Y
3. 计算损失，衡量预测值与实际值之间的差距﻿
4. 反向传播，来计算损失函数对 W, b 的梯度 dW ，db，﻿
5. 然后通过随机梯度下降等算法来进行梯度更新，重复第二到第四步直到损失函数收敛到最小。﻿

其中第一步 权重的初始化 对模型的训练速度和准确性起着重要的作用，所以需要正确地进行初始化。﻿

---

### 下面两种方式，会给模型的训练带来一些问题。﻿

**1. 将所有权重初始化为零﻿**

会使模型相当于是一个线性模型，因为如果将权重初始化为零，那么损失函数对每个 w 的梯度都会是一样的，这样在接下来的迭代中，同一层内所有神经元的梯度相同，梯度更新也相同，所有的权重也都会具有相同的值，这样的神经网络和一个线性模型的效果差不多。（将 biases 设为零不会引起多大的麻烦，即使 bias 为 0，每个神经元的值也是不同的。）﻿


**2. 随机初始化﻿**

将权重进行随机初始化，使其服从标准正态分布 （ np.random.randn(size_l, size_l-1) ﻿ ）
在训练深度神经网络时可能会造成两个问题，**梯度消失和梯度爆炸**。﻿


- **梯度消失﻿**

是指在深度神经网络的反向传播过程中，随着越向回传播，权重的梯度变得越来越小，越靠前的层训练的越慢，导致结果收敛的很慢，损失函数的优化很慢，有的甚至会终止网络的训练。﻿

**解决方案有：**

	- Hessian Free Optimizer With Structural Dumping，
	- Leaky Integration Units，
	- Vanishing Gradient Regularization，
	- Long Short-Term Memory，
	- Gated Recurrent Unit，
	- Orthogonal initialization

- **梯度爆炸﻿**

和梯度消失相反，例如当你有很大的权重，和很小的激活函数值时，这样的权重沿着神经网络一层一层的乘起来，会使损失有很大的改变，梯度也变得很大，也就是 W 的变化（W - ⍺* dW）会是很大的一步，这可能导致在最小值周围一直振荡，一次一次地越过最佳值，模型可能一直也学不到最佳。爆炸梯度还有一个影响是可能发生数值溢出，导致计算不正确，出现 NaN，loss 也出现 NaN 的结果。

**解决方案有：**

	- Truncated Backpropagation Through Time (TBPTT)，
	- L1 and L2 Penalty On The Recurrent Weights，
	- Teacher Forcing，
	- Clipping Gradients，
	- Echo State Networks

相关文章 [梯度消失问题与如何选择激活函数](https://www.jianshu.com/p/c663542f56fe)

---

### 梯度消失和爆炸的应对方案有很多，﻿本文主要看权重矩阵的初始化

对于深度网络，我们可以根据不同的非线性激活函数用不同方法来初始化权重。﻿

也就是**初始化时，并不是服从标准正态分布，而是让 w 服从方差为 k/n 的正态分布**，其中 k 因激活函数而不同。﻿这些方法并不能完全解决梯度爆炸/消失的问题，但在很大程度上可以缓解。


- **对于 RELU(z)**，用下面这个式子乘以随机生成的 w，也叫做 He Initialization：﻿

![](https://upload-images.jianshu.io/upload_images/1667471-cba9a7e636657817.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- **对于 tanh(z)**，用 Xavier 初始化方法，即用下面这个式子乘以随机生成的 w，和上一个的区别就是 k 等于 1 而不是 2。﻿

![](https://upload-images.jianshu.io/upload_images/1667471-bdcdd9c781a5d044.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在 TensorFlow  中：

```
W = tf.get_variable('W', [dims], tf.contrib.layers.xavier_initializer()) 
```

- **还有一种是用下面这个式子乘以 w**：

![](https://upload-images.jianshu.io/upload_images/1667471-d68c4e638ce98bce.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

上面这几个初始化方法可以减少梯度爆炸或消失， 通过这些方式，w 既不会比 1 大很多，也不会比 1 小很多，所以梯度不会很快地消失或爆炸，可以避免收敛太慢，也不会一直在最小值附近震荡。 



---

学习资料：
https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
https://www.leiphone.com/news/201703/3qMp45aQtbxTdzmK.html

---

推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的：
[入门问题][TensorFlow][深度学习][强化学习][神经网络][机器学习][自然语言处理][聊天机器人]

GAN 的 keras 实现

本文结构：

- 什么是 GAN？
- 优点？
- keras 例子？

---

#### 什么是 GAN？

GAN，全称为 Generative Adversarial Nets，直译为生成式对抗网络，是一种非监督式模型。

一种**应用**是生成在原始数据集中不存在的但是却比较合理的数据，还可以拓展一张图片，生成下一帧影像，由简单几笔生成一幅画：

![](http://upload-images.jianshu.io/upload_images/1667471-b015b7aab8938ee1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**模型：**

主要有两部分：

**The Generative Model：**通过输入任意随机数据，尝试生成一些真实的东西（曲线，图像，声音，文本，...）

**The Discriminative Model：**试图判定哪些是虚假的数据，来减小对真实数据的误报。

![](http://upload-images.jianshu.io/upload_images/1667471-f1f2f317df350299.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

#### 优点：

Markov chains are never needed
避免了计算复杂度特别高的过程，直接进行采样和推断，应用效率相应提高。

a wide variety of functions can be incorporated into the model
针对不同的任务就可以设计不同类型的损失函数。

can represent very sharp, even degenerate distributions

---

#### Keras 例子：

任务：生成 sin 曲线。

```
%matplotlib inline
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
from keras.models import Model
from keras.layers import Input, Reshape
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard
```

**1. Generative model:**

输入：noise data
输出：尝试生成真实的 sin 数据

```
def get_generative(G_in, dense_dim=200, out_dim=50, lr=1e-3):
    x = Dense(dense_dim)(G_in)
    x = Activation('tanh')(x)
    G_out = Dense(out_dim, activation='tanh')(x)
    G = Model(G_in, G_out)
    opt = SGD(lr=lr)
    G.compile(loss='binary_crossentropy', optimizer=opt)
    return G, G_out
```

**2. Discriminative model：**

输出：识别此数据是真实的，还是由 Generative model 生成的

```
def get_discriminative(D_in, lr=1e-3, drate=.25, n_channels=50, conv_sz=5, leak=.2):
    x = Reshape((-1, 1))(D_in)
    x = Conv1D(n_channels, conv_sz, activation='relu')(x)
    x = Dropout(drate)(x)
    x = Flatten()(x)
    x = Dense(n_channels)(x)
    D_out = Dense(2, activation='sigmoid')(x)
    D = Model(D_in, D_out)
    dopt = Adam(lr=lr)
    D.compile(loss='binary_crossentropy', optimizer=dopt)
    return D, D_out
```

**3. chain the two models into a GAN：**

set_trainability 的作用是每次训练 generator 时要冻住 discriminator。

```
def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
        
def make_gan(GAN_in, G, D):
    set_trainability(D, False)
    x = G(GAN_in)
    GAN_out = D(x)
    GAN = Model(GAN_in, GAN_out)
    GAN.compile(loss='binary_crossentropy', optimizer=G.optimizer)
    return GAN, GAN_out
```

**4. Training：**

交替训练 discriminator  和 chained GAN，在训练 chained GAN 时要冻住 discriminator  的参数：

![](http://upload-images.jianshu.io/upload_images/1667471-970cfe6d36170c01.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
def sample_noise(G, noise_dim=10, n_samples=10000):
    X = np.random.uniform(0, 1, size=[n_samples, noise_dim])
    y = np.zeros((n_samples, 2))
    y[:, 1] = 1
    return X, y

def train(GAN, G, D, epochs=500, n_samples=10000, noise_dim=10, batch_size=32, verbose=False, v_freq=50):
    d_loss = []
    g_loss = []
    e_range = range(epochs)
    if verbose:
        e_range = tqdm(e_range)
    for epoch in e_range:
        X, y = sample_data_and_gen(G, n_samples=n_samples, noise_dim=noise_dim)
        set_trainability(D, True)
        d_loss.append(D.train_on_batch(X, y))
        
        X, y = sample_noise(G, n_samples=n_samples, noise_dim=noise_dim)
        set_trainability(D, False)
        g_loss.append(GAN.train_on_batch(X, y))
        if verbose and (epoch + 1) % v_freq == 0:
            print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, g_loss[-1], d_loss[-1]))
    return d_loss, g_loss

d_loss, g_loss = train(GAN, G, D, verbose=True)
```

**5. Results：**

```
N_VIEWED_SAMPLES = 2
data_and_gen, _ = sample_data_and_gen(G, n_samples=N_VIEWED_SAMPLES)
pd.DataFrame(np.transpose(data_and_gen[N_VIEWED_SAMPLES:])).rolling(5).mean()[5:].plot()
```

![](http://upload-images.jianshu.io/upload_images/1667471-ea6e985af7dd4ce6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

学习资料：
https://arxiv.org/pdf/1406.2661.pdf
http://www.rricard.me/machine/learning/generative/adversarial/networks/2017/04/05/gans-part1.html
http://www.rricard.me/machine/learning/generative/adversarial/networks/keras/tensorflow/2017/04/05/gans-part2.html

---

推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的：
[入门问题][TensorFlow][深度学习][强化学习][神经网络][机器学习][自然语言处理][聊天机器人]
双向 LSTM

本文结构：

- 为什么用双向 LSTM
- 什么是双向 LSTM
- 例子

---

#### 为什么用双向 LSTM？

单向的 RNN，是根据前面的信息推出后面的，但有时候只看前面的词是不够的，
例如，

我今天不舒服，我打算____一天。

只根据‘不舒服‘，可能推出我打算‘去医院‘，‘睡觉‘，‘请假‘等等，但如果加上后面的‘一天‘，能选择的范围就变小了，‘去医院‘这种就不能选了，而‘请假‘‘休息‘之类的被选择概率就会更大。

---

#### 什么是双向 LSTM？

双向卷积神经网络的隐藏层要保存两个值， A 参与正向计算， A' 参与反向计算。
最终的输出值 y 取决于 A 和 A'：

![](http://upload-images.jianshu.io/upload_images/1667471-ad054c3a8b703f28.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

即正向计算时，隐藏层的 s_t 与 s_t－1 有关；反向计算时，隐藏层的 s_t 与 s_t＋1 有关：

![](http://upload-images.jianshu.io/upload_images/1667471-b6dddc4e9d2b5fd4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](http://upload-images.jianshu.io/upload_images/1667471-d2e41409e1337748.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在某些任务中，双向的 lstm 要比单向的 lstm 的表现要好：

![](http://upload-images.jianshu.io/upload_images/1667471-bba99f50ee3d9784.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

#### 例子

下面是一个 keras 实现的 双向LSTM 应用的小例子，任务是对序列进行分类，
例如如下 10 个随机数：

`0.63144003 0.29414551 0.91587952 0.95189228 0.32195638 0.60742236 0.83895793 0.18023048 0.84762691 0.29165514`

累加值超过设定好的阈值时可标记为 1，否则为 0，例如阈值为 2.5，则上述输入的结果为：

`0 0 0 1 1 1 1 1 1 1`

和单向 LSTM 的区别是用到 Bidirectional：
`model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1)))`


```
from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

# create a sequence classification instance
def get_sequence(n_timesteps):
	# create a sequence of random numbers in [0,1]
	X = array([random() for _ in range(n_timesteps)])
	# calculate cut-off value to change class values
	limit = n_timesteps/4.0
	# determine the class outcome for each item in cumulative sequence
	y = array([0 if x < limit else 1 for x in cumsum(X)])
	# reshape input and output data to be suitable for LSTMs
	X = X.reshape(1, n_timesteps, 1)
	y = y.reshape(1, n_timesteps, 1)
	return X, y

# define problem properties
n_timesteps = 10

# define LSTM
model = Sequential()
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, 1)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# train LSTM
for epoch in range(1000):
	# generate new random sequence
	X,y = get_sequence(n_timesteps)
	# fit model for one epoch on this sequence
	model.fit(X, y, epochs=1, batch_size=1, verbose=2)
	
# evaluate LSTM
X,y = get_sequence(n_timesteps)
yhat = model.predict_classes(X, verbose=0)
for i in range(n_timesteps):
	print('Expected:', y[0, i], 'Predicted', yhat[0, i])

```

---

学习资料：
https://zybuluo.com/hanbingtao/note/541458
https://maxwell.ict.griffith.edu.au/spl/publications/papers/ieeesp97_schuster.pdf
http://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/

---
推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的：
[入门问题][TensorFlow][深度学习][强化学习][神经网络][机器学习][自然语言处理][聊天机器人]
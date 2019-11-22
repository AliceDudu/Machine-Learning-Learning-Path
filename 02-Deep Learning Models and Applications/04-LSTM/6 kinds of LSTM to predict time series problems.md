

6 种用 LSTM 做时间序列预测的模型结构 - Keras 实现

**LSTM(Long Short Term Memory Network)长短时记忆网络**，是一种改进之后的循环神经网络，可以解决 RNN 无法处理长距离的依赖的问题，在时间序列预测问题上面也有广泛的应用。

![](https://upload-images.jianshu.io/upload_images/1667471-4e5a0a8efd1881b2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

今天我们根据问题的输入输出模式划分，来看一下几种时间序列问题所对应的 LSTM 模型结构如何实现。

![](https://upload-images.jianshu.io/upload_images/1667471-59230a4fdbcd874c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

### 1. Univariate

![](https://upload-images.jianshu.io/upload_images/1667471-e6d55f919ce05224.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Univariate 是指： 

input 为多个时间步，
output 为一个时间的问题。

**数例：**

```
训练集：
X,			y
10, 20, 30		40
20, 30, 40		50
30, 40, 50		60
…


预测输入：
X，
70, 80, 90
```

**模型的 Keras 代码：**

```
# define model【Vanilla LSTM】

model = Sequential()
model.add( LSTM(50,  activation='relu',  input_shape = (n_steps, n_features)) )
model.add( Dense(1) )
model.compile(optimizer='adam', loss='mse')

n_steps = 3
n_features = 1
```

其中：

`n_steps` 为输入的 X 每次考虑几个**时间步**
`n_features` 为每个时间步的**序列数**

这个是最基本的模型结构，我们后面几种模型会和这个进行比较。

---

### 2. Multiple Input

![](https://upload-images.jianshu.io/upload_images/1667471-8e2bd2fb83d022f6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


Multiple Input 是指： 

input 为多个序列，
output 为一个序列的问题。

**数例：**

```
训练集：
X，       y
[[10 15]
 [20 25]
 [30 35]] 65
[[20 25]
 [30 35]
 [40 45]] 85
[[30 35]
 [40 45]
 [50 55]] 105
[[40 45]
 [50 55]
 [60 65]] 125
…


预测输入：
X，
80,	 85
90,	 95
100,     105
```

即数据样式为：

```
in_seq1： [10, 20, 30, 40, 50, 60, 70, 80, 90]
in_seq2： [15, 25, 35, 45, 55, 65, 75, 85, 95]

out_seq： [in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))]
```

**模型的 Keras 代码：**

```
# define model【Vanilla LSTM】
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

n_steps = 3
# 此例中 n features = 2，因为输入有两个并行序列
n_features = X.shape[2]    
```

其中：

`n_steps` 为输入的 X 每次考虑几个时间步
`n_features` 此例中 = 2，因为输入有**两个并行序列**

**和 Univariate 相比：**

模型的结构代码是一样的，只是在 `n_features = X.shape[2]`，而不是 1.


---

### 3. Multiple Parallel

![](https://upload-images.jianshu.io/upload_images/1667471-a4746904b9c75cfc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


Multiple Parallel 是指： 

input 为多个序列，
output 也是多个序列的问题。

**数例：**

```
训练集：
X,			y
[[10 15 25]
 [20 25 45]
 [30 35 65]] [40 45 85]
[[20 25 45]
 [30 35 65]
 [40 45 85]] [ 50  55 105]
[[ 30  35  65]
 [ 40  45  85]
 [ 50  55 105]] [ 60  65 125]
[[ 40  45  85]
 [ 50  55 105]
 [ 60  65 125]] [ 70  75 145]
…


预测输入：
X，
70, 75, 145
80, 85, 165
90, 95, 185
```

**模型的 Keras 代码：**

```
# define model【Vanilla LSTM】
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')

n_steps = 3
# 此例中 n features = 3，因为输入有3个并行序列
n_features = X.shape[2]       
```

其中：

`n_steps` 为输入的 X 每次考虑几个时间步
`n_features` 此例中 = 3，因为输入有 3 个并行序列

**和 Univariate 相比：**

模型结构的定义中，多了一个 `return_sequences=True`，即返回的是序列，
输出为 `Dense(n_features)`，而不是 1.

---

### 4. Multi-Step

![](https://upload-images.jianshu.io/upload_images/1667471-869efd8de0f12dab.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Multi-Step 是指： 

input 为多个时间步，
output 也是**多个时间步**的问题。

**数例：**

```
训练集：
X,			y
[10 20 30] [40 50]
[20 30 40] [50 60]
[30 40 50] [60 70]
[40 50 60] [70 80]
…


预测输入：
X，
[70, 80, 90]
```

**模型的 Keras 代码：**

```
# define model【Vanilla LSTM】
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

n_steps_in, n_steps_out = 3, 2
n_features = 1     
```

其中：

`n_steps_in` 为输入的 X 每次考虑几个时间步
`n_steps_out` 为输出的 y 每次考虑几个时间步
`n_features` 为输入有几个序列

**和 Univariate 相比：**

模型结构的定义中，多了一个 `return_sequences=True`，即返回的是序列，
而且 `input_shape=(n_steps_in, n_features)` 中有代表输入时间步数的 `n_steps_in`，
输出为 `Dense(n_steps_out)`，代表输出的 y 每次考虑几个时间步.

**当然这个问题还可以用 Encoder-Decoder 结构实现：**

```
# define model【Encoder-Decoder Model】
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
```

---

### 5. Multivariate Multi-Step

![](https://upload-images.jianshu.io/upload_images/1667471-8839412a096c4dde.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Multivariate Multi-Step 是指： 

input 为多个序列，
output 为多个时间步的问题。

**数例：**

```
训练集：
X,			y
[[10 15]
 [20 25]
 [30 35]] [65 
	      85]
[[20 25]
 [30 35]
 [40 45]] [ 85
	       105]
[[30 35]
 [40 45]
 [50 55]] [105 
         125]
…


预测输入：
X，
[40 45]
 [50 55]
 [60 65]
```

**模型的 Keras 代码：**

```
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

n_steps_in, n_steps_out = 3, 2
# 此例中 n features = 2，因为输入有2个并行序列  
n_features = X.shape[2]        
```

其中：

`n_steps_in` 为输入的 X 每次考虑几个时间步
`n_steps_out` 为输出的 y 每次考虑几个时间步
`n_features` 为输入有几个序列，此例中 = 2，因为输入有 2 个并行序列  

**和 Univariate 相比：**

模型结构的定义中，多了一个 `return_sequences=True`，即返回的是序列，
而且 `input_shape=(n_steps_in, n_features)` 中有代表输入时间步数的 `n_steps_in`，
输出为 `Dense(n_steps_out)`，代表输出的 y 每次考虑几个时间步，
另外 `n_features = X.shape[2]`，而不是 1，
相当于是 Multivariate 和 Multi-Step 的结构组合起来。

---

### 6. Multiple Parallel Input & Multi-Step Output

![](https://upload-images.jianshu.io/upload_images/1667471-b2ff12f2cd78502a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Multiple Parallel Input & Multi-Step Output 是指：
 
input 为多个序列，
output 也是多个序列 & 多个时间步的问题。

**数例：**

```
训练集：
X,			y
[[10 15 25]
 [20 25 45]
 [30 35 65]] [[ 40  45  85]
 	      [ 50  55 105]]
[[20 25 45]
 [30 35 65]
 [40 45 85]] [[ 50  55 105]
 	      [ 60  65 125]]
[[ 30  35  65]
 [ 40  45  85]
 [ 50  55 105]] [[ 60  65 125]
 	         [ 70  75 145]]
…


预测输入：
X，
[[ 40  45  85]
 [ 50  55 105]
 [ 60  65 125]]
```

**模型的 Keras 代码：**

```
# define model【Encoder-Decoder model】
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')

n_steps_in, n_steps_out = 3, 2
# 此例中 n features = 3，因为输入有3个并行序列   
n_features = X.shape[2]       
```

其中：

`n_steps_in` 为输入的 X 每次考虑几个时间步
`n_steps_out` 为输出的 y 每次考虑几个时间步
`n_features` 为输入有几个序列

**这里我们和 Multi-Step 的 Encoder-Decoder 相比：**

二者的模型结构，只是在最后的输出层参数不同，
`TimeDistributed(Dense(n_features))` 而不是 `Dense(1)`。

---

好啦，这几种时间序列的输入输出模式所对应的代码结构就是这样，如果您还有更有趣的，欢迎补充！

---

大家好！
我是 **不会停的蜗牛 Alice，**
喜欢人工智能，没事儿写写机器学习干货，
欢迎关注我！

---

推荐阅读历史技术博文链接汇总
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的：
[入门问题][TensorFlow][深度学习][强化学习][神经网络][机器学习][自然语言处理][聊天机器人]







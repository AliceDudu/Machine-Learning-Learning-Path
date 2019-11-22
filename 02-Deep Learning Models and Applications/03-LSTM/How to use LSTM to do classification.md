用 LSTM 来做一个分类小问题

**用一个简单的例子来看看 LSTM 在 tensorflow 里是如何做分类问题的。**

这个例子特别简单，就是一个长度为 20 的二进制串，数出其中 1 的个数，简单到用一个 for 就能搞定的事情，来看看 LSTM 是如何做到的。

大家可以先在这里停一下，看看你有什么想法呢。

---

```python
import numpy as np
from random import shuffle
```

**input 一共有 2^20 种组合，就生成这么多的数据**

```python
train_input = ['{0:020b}'.format(i) for i in range(2**20)]
shuffle(train_input)
train_input = [map(int,i) for i in train_input]
```

train_input：
[1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
[0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1]

**把每一个 input 转化成 tensor 的形式**
在 `dimensions ＝ [batch_size, sequence_length, input_dimension]` 中，
`sequence_length ＝ 20 and input_dimension ＝ 1，`
每个 input 变成了 A list of 20 lists 

```python
ti  = []
for i in train_input:
    temp_list = []    
    for j in i:
            temp_list.append([j])            
    ti.append( np.array(temp_list) )
    
train_input = ti
```

train_input ：
[[1][0][0][0][1][1][1][0][1][0][0][0][0][1][0][0][0][1][0][0]]

**生成实际的 output 数据**

```python
train_output = []
 
for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count+=1
    temp_list = ([0]*21)
    temp_list[count]=1
    train_output.append(temp_list)
```

train_output：在第几个位置上有一个 1 ，说明 input 里面就有几个 1，长度为 21
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**取 0.9% 为训练数据，另外的为测试数据**

```python
NUM_EXAMPLES = 10000
test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:] #everything beyond 10,000
 
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES] #till 10,000
```

**定义两个变量**
其中 data 的维度 ＝ [Batch Size, Sequence Length, Input Dimension]

```python
data = tf.placeholder(tf.float32, [None, 20,1])
target = tf.placeholder(tf.float32, [None, 21])
```

**定义 hidden dimension ＝ 24**
太多会 overfitting，太少效果不好，可以调节看变化。
模型用 LSTM，这里用的 tf 1.0.0 的 version

```python
num_hidden = 24
# cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
```

**用 val 来存这个 output**

```python
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
```

**变换一下维度，并取 val 的最后一个为 last**

```python
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)
```

![](http://upload-images.jianshu.io/upload_images/1667471-137226e40d8a9a94.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**定义 weight 和 bias**

```python
weight = tf.Variable(tf.truncated_normal( [num_hidden, int(target.get_shape()[1])] ))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
```

**再作用上 softmax 得到 prediction**

```python
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
```
**用 cross_entropy 来做 cost function，目标是使它最小化，选用 AdamOptimizer**

```python
cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)
```

**定义一下 error 的形式，就是预测和实际有多少个位置不一样**

```python
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
```


![](http://upload-images.jianshu.io/upload_images/1667471-8185cd84ec5fdb47.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

**前面定义完模型和变量，这里开始启动 session**

```python
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)
```

迭代 600 次就可以达到 0.3% 的 error 了

```python
batch_size = 1000
no_of_batches = int(len(train_input)) / batch_size
epoch = 600
```


```python
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr += batch_size
        sess.run(minimize,{data: inp, target: out})
    print "Epoch ",str(i)
    
incorrect = sess.run(error,{data: test_input, target: test_output})

print sess.run(prediction, {data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]})
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))

sess.close()
```

最后的结果：

```python
[[  2.80220238e-08   3.24575727e-10   5.68697936e-11   3.57573054e-10
    9.62089857e-08   1.30921896e-08   2.14473985e-08   5.21751364e-10
    2.29034747e-08   8.47907577e-10   3.60394756e-06   2.30961153e-03
    9.82593179e-01   1.50928665e-02   4.23395448e-07   1.06428047e-07
    6.70640388e-09   1.78888765e-10   3.22445395e-08   3.09186134e-08
    3.70296416e-09]]
    
Epoch 600 error 0.3%
```

学习资料：
http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/

---

推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的
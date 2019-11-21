用 Tensorflow 建立 CNN

稍稍乱入的CNN，本文依然是学习[周莫烦视频的笔记。](https://www.youtube.com/watch?v=tjcgL5RIdTM&index=19&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8)

还有 google 在 udacity 上的 [CNN 教程。](https://classroom.udacity.com/courses/ud730/lessons/6377263405/concepts/63796332430923)



**CNN(Convolutional Neural Networks) 卷积神经网络**简单讲就是把一个图片的数据传递给CNN，原涂层是由RGB组成，然后CNN把它的厚度加厚，长宽变小，每做一层都这样被拉长，最后形成一个分类器：


![](http://upload-images.jianshu.io/upload_images/1667471-3d06daf10515c57f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](http://upload-images.jianshu.io/upload_images/1667471-82842bb1e4e5c927.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


如果想要分成十类的话，那么就会有0到9这十个位置，这个数据属于哪一类就在哪个位置上是1，而在其它位置上为零。

在 RGB 这个层，每一次把一块核心抽出来，然后厚度加厚，长宽变小，形成分类器：

![](http://upload-images.jianshu.io/upload_images/1667471-52f2f5a4a6699bdf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


**在 CNN 中有几个重要的概念：**
- stride
- padding
- pooling

**stride，**就是每跨多少步抽取信息。每一块抽取一部分信息，长宽就缩减，但是厚度增加。抽取的各个小块儿，再把它们合并起来，就变成一个压缩后的立方体。

**padding，**抽取的方式有两种，一种是抽取后的长和宽缩减，另一种是抽取后的长和宽和原来的一样。

**pooling，**就是当跨步比较大的时候，它会漏掉一些重要的信息，为了解决这样的问题，就加上一层叫pooling，事先把这些必要的信息存储起来，然后再变成压缩后的层：


![](http://upload-images.jianshu.io/upload_images/1667471-7fda4b9d7141db3c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**patch, **就是小方块的长宽的像素，in size 是image的厚度为1，out size是输出的厚度为32:

![](http://upload-images.jianshu.io/upload_images/1667471-a1c239ca2c056a28.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**CNN的结构，**分析一张图片时，先放一个CNN的图层，再把这个图层进行一个pooling。这样可以比较好的保持信息，之后再加第二层的CNN和pooling。



导入一个图片之后，先是有它的RGB三个图层，然后把像素块缩小变厚。本来有三个厚度，然后把它变成八个厚度，它的长宽在不断的减小，最后把它们连接在一起：


![](http://upload-images.jianshu.io/upload_images/1667471-0d067b4d78aeb0d1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)





**下面就是用 tensorflow 构建一个 CNN 的代码，**
里面主要有4个layer，分别是:
1. convolutional layer1 + max pooling;
2. convolutional layer2 + max pooling;
3. fully connected layer1 + dropout;
4. fully connected layer2 to prediction.



``` python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

# 产生随机变量，符合 normal 分布
# 传递 shape 就可以返回weight和bias的变量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)	
    return tf.Variable(initial)							

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 定义2维的 convolutional 图层
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    # strides 就是跨多大步抽取信息
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')		

# 定义 pooling 图层
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    # 用pooling对付跨步大丢失信息问题
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')		

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) 		# 784＝28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])			# 最后一个1表示数据是黑白的
# print(x_image.shape)  # [n_samples, 28,28,1]

## 1. conv1 layer ##
#  把x_image的厚度1加厚变成了32
W_conv1 = weight_variable([5, 5, 1, 32]) 				# patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
# 构建第一个convolutional层，外面再加一个非线性化的处理relu
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 			# output size 28x28x32
# 经过pooling后，长宽缩小为14x14
h_pool1 = max_pool_2x2(h_conv1)                                     # output size 14x14x32

## 2. conv2 layer ##
# 把厚度32加厚变成了64
W_conv2 = weight_variable([5,5, 32, 64]) 				# patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
# 构建第二个convolutional层
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) 			# output size 14x14x64
# 经过pooling后，长宽缩小为7x7
h_pool2 = max_pool_2x2(h_conv2)                                     # output size 7x7x64

## 3. func1 layer ##
# 飞的更高变成1024
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
# 把pooling后的结果变平
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## 4. func2 layer ##
# 最后一层，输入1024，输出size 10，用 softmax 计算概率进行分类的处理
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
```
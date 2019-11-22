神经网络 之 感知器的概念和实现

本文结构：

1. 什么是感知器
2. 有什么用
3. 代码实现

---

###1. 什么是感知器


如下图，这个神经网络中，每个圆圈都是一个神经元，神经元也叫做感知器

![](http://upload-images.jianshu.io/upload_images/1667471-24f3e1823973e1ba.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

只有一个隐藏层的神经网络就能拟合任何一个函数，但是它需要很多很多的神经元。
而深层网络用相对少的神经元就能拟合同样的函数，但是层数增加了，不太容易训练，需要大量的数据。
为了拟合一个函数，可以使用一个浅而宽的网络，也可以使用一个深而窄的网络，后者更节约资源。


**下图单挑出一个感知器来看：**
向它输入 inputs，经过 加权 求和，再作用上激活函数后，得到一个输出值

![](http://upload-images.jianshu.io/upload_images/1667471-e853dcb5c062ea07.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

感知器的激活函数可以有很多选择，关于激活函数可以看 [常用激活函数比较](http://www.jianshu.com/p/22d9720dbf1a)


---

###2. 有什么用

用感知器可以实现 and 函数，or 函数，还可以拟合任何的线性函数，任何线性分类或线性回归问题都可以用感知器来解决。

但是，感知器却不能实现异或运算，如下图所示，异或运算不是线性的，无法用一条直线把 0 和 1 分开。

![xor](http://upload-images.jianshu.io/upload_images/1667471-1b39373f24172e01.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


**训练权重和偏置的算法如下：**

![](http://upload-images.jianshu.io/upload_images/1667471-067075da40037f8f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


![](http://upload-images.jianshu.io/upload_images/1667471-24376c679ec40dc5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

其中，t 是训练样本的实际值，y 是感知器的输出值，即由 f 计算出来的。eta 称为学习速率，是个常数，作用是控制每一步调整权的幅度。

---

###3. 代码实现

####［main］

先训练and感知器
```
and_perception = train_and_perceptron()
```

得到训练后获得的权重和偏置
```
print and_perception	
```
```
weights	:[0.1, 0.2]
bias	:-0.200000
```

再去测试，看结果是否正确
```
print '1 and 1 = %d' % and_perception.predict([1, 1])
```

其中
####［train_and_perceptron］

先创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
```
    p = Perceptron(2, f)
```

f 为
```
def f(x):
    return 1 if x > 0 else 0
```

输入训练data，迭代10次, 学习速率为0.1
```
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
```

训练data为
```
    input_vecs = [[1,1], [0,0], [1,0], [0,1]]
    labels = [1, 0, 0, 0]
```

关于
####［train］

一共迭代 10 次，每次迭代时，
先计算感知器在当前权重下的输出，然后更新weights
```
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)
```

其中
####［_update_weights］

就是用训练算法里面的两个公式
```
        delta = label - output
        self.weights = map(
            lambda (x, w): w + rate * delta * x,
            zip(input_vec, self.weights) )
        self.bias += rate * delta
```

![](http://upload-images.jianshu.io/upload_images/1667471-00eb84a64e743cf1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


当
####［predict］

就用感知器的函数 f：

![](http://upload-images.jianshu.io/upload_images/1667471-7622f2e0be50a345.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
        return self.activator(
            reduce(lambda a, b: a + b,
                   map(lambda (x, w): x * w,  
                       zip(input_vec, self.weights))
                , 0.0) + self.bias)
```

![](http://upload-images.jianshu.io/upload_images/1667471-ab0bb34b36e19843.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

完整代码：

```
#!/usr/bin/python
#-*-coding:utf-8 -*-

class Perceptron(object):
    def __init__(self, input_num, activator):
        '''
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为double -> double
        '''
        self.activator = activator
        # 权重向量初始化为0
        self.weights = [0.0 for _ in range(input_num)]
        # 偏置项初始化为0
        self.bias = 0.0
        
    def __str__(self):
        '''
        打印学习到的权重、偏置项
        '''
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)
        
    def predict(self, input_vec):
        '''
        输入向量，输出感知器的计算结果
        '''
        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
        # 最后利用reduce求和
        return self.activator(
            reduce(lambda a, b: a + b,
                   map(lambda (x, w): x * w,  
                       zip(input_vec, self.weights))
                , 0.0) + self.bias)
                
    def train(self, input_vecs, labels, iteration, rate):
        '''
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)
            
    def _one_iteration(self, input_vecs, labels, rate):
        '''
        一次迭代，把所有的训练数据过一遍
        '''
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples = zip(input_vecs, labels)
        # 对每个样本，按照感知器规则更新权重
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)
            
    def _update_weights(self, input_vec, output, label, rate):
        '''
        按照感知器规则更新权重
        '''
        # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        delta = label - output
        self.weights = map(
            lambda (x, w): w + rate * delta * x,
            zip(input_vec, self.weights) )
        # 更新bias
        self.bias += rate * delta

def f(x):
    '''
    定义激活函数f
    '''
    return 1 if x > 0 else 0
    
def get_training_dataset():
    '''
    基于and真值表构建训练数据
    '''
    # 构建训练数据
    # 输入向量列表
    input_vecs = [[1,1], [0,0], [1,0], [0,1]]
    # 期望的输出列表，注意要与输入一一对应
    # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
    labels = [1, 0, 0, 0]
    return input_vecs, labels   
     
def train_and_perceptron():
    '''
    使用and真值表训练感知器
    '''
    # 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
    p = Perceptron(2, f)
    # 训练，迭代10轮, 学习速率为0.1
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    #返回训练好的感知器
    return p
    
if __name__ == '__main__': 
    # 训练and感知器
    and_perception = train_and_perceptron()
    # 打印训练获得的权重
    print and_perception
    # 测试
    print '1 and 1 = %d' % and_perception.predict([1, 1])
    print '0 and 0 = %d' % and_perception.predict([0, 0])
    print '1 and 0 = %d' % and_perception.predict([1, 0])
    print '0 and 1 = %d' % and_perception.predict([0, 1])
```

参考资料：
https://www.zybuluo.com/hanbingtao/note/433855

---
推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
也许可以找到你想要的

我是 *不会停的蜗牛* Alice
85后全职主妇
喜欢人工智能，行动派
创造力，思考力，学习力提升修炼进行中
欢迎您的喜欢，关注和评论！

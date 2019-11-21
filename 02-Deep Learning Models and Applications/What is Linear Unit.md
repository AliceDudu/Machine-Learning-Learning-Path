
神经网络 之 线性单元

**本文结构：**

1. 什么是线性单元
2. 有什么用
3. 代码实现

---

###1. 什么是线性单元

线性单元和感知器的区别就是在激活函数：

![](http://upload-images.jianshu.io/upload_images/1667471-f8626ad40e4c1353.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

感知器的 f 是阶越函数：

![](http://upload-images.jianshu.io/upload_images/1667471-686ee0338cb2bc46.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

线性单元的激活函数是线性的：

![](http://upload-images.jianshu.io/upload_images/1667471-890ea9159289490f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)




所以线性模型的公式如下：


![](http://upload-images.jianshu.io/upload_images/1667471-4d28ef8bd83cb666.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



![](http://upload-images.jianshu.io/upload_images/1667471-172ebcf3bdef6d05.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

###2. 有什么用

感知器存在一个问题，就是遇到线性不可分的数据时，就可能无法收敛，所以要使用一个可导的线性函数来替代阶跃函数，即线性单元，这样就会收敛到一个最佳的近似上。

![](http://upload-images.jianshu.io/upload_images/1667471-6bc5801355a37a96.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


###3. 代码实现



**1. 继承Perceptron，初始化线性单元**

```python
from perceptron import Perceptron
#定义激活函数f
f = lambda x: x
class LinearUnit(Perceptron):
    def __init__(self, input_num):
        '''初始化线性单元，设置输入参数的个数'''
        Perceptron.__init__(self, input_num, f)
```

**2. 定义一个线性单元, 调用 `train_linear_unit` 进行训练**
- 打印训练获得的权重
- 输入参数值 [3.4] 测试一下预测值

```python
if __name__ == '__main__': 
    '''训练线性单元'''
    linear_unit = train_linear_unit()
    # 打印训练获得的权重
    print linear_unit
    # 测试
    print 'Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4])
    print 'Work 15 years, monthly salary = %.2f' % linear_unit.predict([15])
    print 'Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5])
    print 'Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3])
```

- 其中训练的过程就是：
- 获得训练数据，
- 设定迭代次数，学习速率等参数
- 再返回训练好的线性单元

```python
def train_linear_unit():
    '''
    使用数据训练线性单元
    '''
    # 创建感知器，输入参数的特征数为1（工作年限）
    lu = LinearUnit(1)
    # 训练，迭代10轮, 学习速率为0.01
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    #返回训练好的线性单元
    return lu
```


**完整代码**

```python
from perceptron import Perceptron
#定义激活函数f
f = lambda x: x
class LinearUnit(Perceptron):
    def __init__(self, input_num):
        '''初始化线性单元，设置输入参数的个数'''
        Perceptron.__init__(self, input_num, f)


def get_training_dataset():
    '''
    捏造5个人的收入数据
    '''
    # 构建训练数据
    # 输入向量列表，每一项是工作年限
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    # 期望的输出列表，月薪，注意要与输入一一对应
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels    
def train_linear_unit():
    '''
    使用数据训练线性单元
    '''
    # 创建感知器，输入参数的特征数为1（工作年限）
    lu = LinearUnit(1)
    # 训练，迭代10轮, 学习速率为0.01
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    #返回训练好的线性单元
    return lu
if __name__ == '__main__': 
    '''训练线性单元'''
    linear_unit = train_linear_unit()
    # 打印训练获得的权重
    print linear_unit
    # 测试
    print 'Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4])
    print 'Work 15 years, monthly salary = %.2f' % linear_unit.predict([15])
    print 'Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5])
    print 'Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3])
```

学习资料：
https://www.zybuluo.com/hanbingtao/note/448086



---
推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
也许可以找到你想要的

我是 *不会停的蜗牛* Alice
85后全职主妇
喜欢人工智能，行动派
创造力，思考力，学习力提升修炼进行中
欢迎您的喜欢，关注和评论！
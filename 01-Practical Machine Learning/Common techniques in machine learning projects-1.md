机器学习算法应用中常用技巧-1

[参考：Udacity ML纳米学位](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009)

### 1. 取样

数据量很大的时候，想要先选取少量数据来观察一下细节。

``` python
indices = [100,200,300]

# 把sample原来的序号去掉重新分配
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples:"
display(samples)

```

### 2. Split数据

用 ```sklearn.cross_validation.train_test_split``` 将数据分为 train 和 test 集。
[sklearn](http://scikit-learn.org/stable/modules/cross_validation.html#stratified-shuffle-split)

``` python
from sklearn import cross_validation
X = new_data
y = data['Milk']
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.25, random_state = 0)
print len(X_train), len(X_test), len(y_train), len(y_test)

```


### 分离出 Features & Label
有时候原始数据并不指出谁是label，自己判断

``` python
# Store the 'Survived' feature in a new variable and remove it from the dataset
outcomes = full_data['Survived']
data = full_data.drop('Survived', axis=1)

```



### 3. 用 train 来训练模型，用 test 来检验

用 Decision Tree 来做个例子
[sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

``` python
from sklearn import tree
regressor = tree.DecisionTreeRegressor()
regressor = regressor.fit(X_train, y_train)
score = regressor.score(X_test, y_test)
```


### 4. 判断 feature 间的关联程度

``` python
pd.scatter_matrix(data, alpha = 0.3, figsize = (14, 8), diagonal = 'kde');
```

### 5. scaling

当数据不符合正态分布的时候，需要做 scaling 的处理。常用的方法是取log。

``` python
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
```

scaling前后对比图：
![](http://upload-images.jianshu.io/upload_images/1667471-ceadb061c888e8f2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



![](http://upload-images.jianshu.io/upload_images/1667471-380654e7fec8a472.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 6. Outliers

方法之一是 Tukey 方法，小于  Q1 – (1.5 × IQR) 或者大于 Q3 + (1.5 × IQR) 就被看作是outlier。

先把各个 feature 的 outlier 列出来并排好序：
``` python
for feature in log_data.keys():
    Q1 = np.percentile(log_data[feature], 25)
    Q3 = np.percentile(log_data[feature], 75)
    step = 1.5 * (Q3 - Q1)
    print "Outliers for feature '{}':".format(feature)
    print Q1, Q3, step
    display(log_data[~((log_data[feature]>=Q1-step) & (log_data[feature]<=Q3+step))].sort([feature]))
```

再配合 boxplot 观察，到底哪些 outlier 需要被移除：
``` python
plt.figure()
plt.boxplot([log_data.Fresh, log_data.Milk, log_data.Grocery, log_data.Frozen, log_data.Detergents_Paper, log_data.Delicassen], 0, 'gD');
```

![](http://upload-images.jianshu.io/upload_images/1667471-b8d6136d092cea91.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---
[历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)

我是 *不会停的蜗牛* Alice
85后全职主妇
喜欢人工智能，行动派
创造力，思考力，学习力提升修炼进行中
欢迎您的喜欢，关注和评论！

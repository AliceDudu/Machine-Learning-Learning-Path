用 Pipeline 将训练集参数重复应用到测试集

当我们对训练集应用各种预处理操作时（特征标准化、主成分分析等等），
我们都需要对测试集重复利用这些参数。

pipeline 实现了对全部步骤的流式化封装和管理，可以很方便地使参数集在新数据集上被重复使用。

**pipeline 可以用于下面几处：**

- 模块化 Feature Transform，只需写很少的代码就能将新的 Feature 更新到训练集中。
- 自动化 Grid Search，只要预先设定好使用的 Model 和参数的候选，就能自动搜索并记录最佳的 Model。
- 自动化 Ensemble Generation，每隔一段时间将现有最好的 K 个 Model 拿来做 Ensemble。

---

#### 栗子：
问题是要对数据集 Breast Cancer Wisconsin 进行分类，
它包含 569 个样本，第一列 ID，第二列类别(M=恶性肿瘤，B=良性肿瘤)，
第 3-32 列是实数值的特征。

```
from pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                 'breast-cancer-wisconsin/wdbc.data', header=None)
                                 # Breast Cancer Wisconsin dataset

X, y = df.values[:, 2:], df.values[:, 1]

encoder = LabelEncoder()
y = encoder.fit_transform(y)
                    >>> encoder.transform(['M', 'B'])
                    array([1, 0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)
```
 
我们要用 **Pipeline** 对训练集和测试集进行如下操作：

- 先用 `StandardScaler` 对数据集每一列做标准化处理，（是 transformer）
- 再用 `PCA` 将原始的 30 维度特征压缩的 2 维度，（是 transformer）
- 最后再用模型 `LogisticRegression`。（是 Estimator）

**调用 Pipeline 时**，输入由元组构成的列表，每个元组第一个值为变量名，元组第二个元素是 sklearn 中的 transformer 或 Estimator。

注意中间每一步是 **transformer**，即它们必须包含 fit 和 transform 方法，或者  `fit_transform`。
最后一步是一个 **Estimator**，即最后一步模型要有 fit 方法，可以没有 transform 方法。

然后用 **Pipeline.fit**对训练集进行训练，`pipe_lr.fit(X_train, y_train)`
再直接用 **Pipeline.score** 对测试集进行预测并评分 `pipe_lr.score(X_test, y_test)`

```
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

pipe_lr = Pipeline([('sc', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(random_state=1))
                    ])
pipe_lr.fit(X_train, y_train)
print('Test accuracy: %.3f' % pipe_lr.score(X_test, y_test))

                # Test accuracy: 0.947
```

#### 还可以用来选择特征：
例如用 SelectKBest 选择特征，
分类器为 SVM，

```
anova_filter = SelectKBest(f_regression, k=5)
clf = svm.SVC(kernel='linear')

anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
```

完整：

```
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline

# generate some data to play with
X, y = samples_generator.make_classification(
     n_informative=5, n_redundant=0, random_state=42)

# ANOVA SVM-C
anova_filter = SelectKBest(f_regression, k=5)
clf = svm.SVC(kernel='linear')
anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])

anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)

prediction = anova_svm.predict(X)
anova_svm.score(X, y)   
```

#### 当然也可以应用 K-fold cross validation：

```
model = Pipeline(estimators)
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```

完整：

```
# Create a pipeline that standardizes the data then creates a model
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# create pipeline
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lda', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)
# evaluate pipeline
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```


---

#### Pipeline 的工作方式：

当管道 Pipeline 执行 fit 方法时，
首先 StandardScaler 执行 fit 和 transform 方法，
然后将转换后的数据输入给 PCA，
PCA 同样执行 fit 和 transform 方法，
再将数据输入给 LogisticRegression，进行训练。

如下图
![](http://upload-images.jianshu.io/upload_images/1667471-595052437df14870.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

资料：
http://blog.csdn.net/lanchunhui/article/details/50521648
https://dnc1994.com/2016/04/rank-10-percent-in-first-kaggle-competition/

---
推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的：
[入门问题][TensorFlow][深度学习][强化学习][神经网络][机器学习][自然语言处理][聊天机器人]
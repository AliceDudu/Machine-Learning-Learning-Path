机器学习算法应用中常用技巧-2

### 7. 降维－PCA


n_components为降到多少维，用原数据fit后，再用transform转换成降维后的数据。

``` python
from sklearn.decomposition import PCA
pca = PCA(n_components = 6)
pca.fit(good_data)
reduced_data = pca.transform(good_data)
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
```

### 8. 聚类－选择类别数

用 silhouette coefficient 计算每个数据到中心点的距离，-1 (dissimilar) to 1 (similar) 根据这个系数来评价聚类算法的优劣。
``` python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
cluster = KMeans(n_clusters=2, random_state=0).fit(reduced_data)
preds = cluster.predict(reduced_data)
score = silhouette_score(reduced_data, preds)
```
选择分数最大的个数作为聚类的类别数。
![](http://upload-images.jianshu.io/upload_images/1667471-728fe217f355be2f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### 9. 恢复维度

例如数据，先经过 log，又经过 PCA降维， 要恢复回去，先用 pca.inverse_transform，再用 np.exp

``` python
log_centers = pca.inverse_transform(centers)
true_centers = np.exp(log_centers)
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
```

数据变化：
![](http://upload-images.jianshu.io/upload_images/1667471-d04b4882ea1b8b56.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 10. 自定义accuracy
分类问题可以自己写accuracy的函数

``` python
def accuracy_score(truth, pred):
    """ Return accuracy score for input truth and prediction"""
    
    if len(truth)==len(pred):
        return "Accuracy for prediction: {:.2f}%.".format((truth==pred).mean()*100)
    else:
        return "Numbers do not match!"
```



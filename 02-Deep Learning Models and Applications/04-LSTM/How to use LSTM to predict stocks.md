用线性回归和LSTM做股价预测

本文以微软的股价为例，详细注释在代码块里：

---

#### 1. 导入相关的包

```
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
```

---

#### 2. 描述性统计

```
df.head()
```

![](https://upload-images.jianshu.io/upload_images/1667471-5ec994aeef0e072d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
df.describe()
```

![](https://upload-images.jianshu.io/upload_images/1667471-c6faa29eebfcc3fd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

#### 3. 可视化

```
#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#plot
plt.figure(figsize=(16,8))
plt.plot(df['Close Price'], label='Close Price history')
```

![](https://upload-images.jianshu.io/upload_images/1667471-4267bc3a2418c933.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

---

#### 4.Linear Regression

```
from sklearn import preprocessing;
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import linear_model;

def prepare_data(df,forecast_col,forecast_out,test_size):
    label = df[forecast_col].shift(-forecast_out);      # 建立 label，是 forecast_col 这一列的向右错位 forecast_out=5 个位置，多出的是 na
    X = np.array(df[[forecast_col]]);                   # X 为 是 forecast_col 这一列
    X = preprocessing.scale(X)                          # processing X
    X_lately = X[-forecast_out:]                        # X_lately 是 X 的最后 forecast_out 个数，用来预测未来的数据
    X = X[:-forecast_out]                               # X 去掉最后 forecast_out 几个数
    label.dropna(inplace=True);                         # 去掉 na values
    y = np.array(label)                                
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size) 

    response = [X_train,X_test , Y_train, Y_test , X_lately];
    return response;

forecast_col = 'Close Price'                            # 选择 close 这一列
forecast_out = 5                                        # 要预测未来几个时间步 
test_size = 0.2;                                        # test set 的大小

X_train, X_test, Y_train, Y_test , X_lately =prepare_data(df,forecast_col,forecast_out,test_size)

model = linear_model.LinearRegression();              

model.fit(X_train,Y_train);
score = model.score(X_test,Y_test);

score        
# 0.9913674520169482

y_test_predict = learner.predict(X_test)

plt.plot(y_test_predict)
plt.plot(Y_test)
```

![](https://upload-images.jianshu.io/upload_images/1667471-dad52de5773ad37a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
forecast= learner.predict(X_lately)

forecast
# array([112.46087852, 109.20867432, 109.46117455, 108.9258753 ,
       110.10757453])
```

---

#### 5. LSTM

```
# 导入 keras 等相关包
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# 选取 date 和 close 两列
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close Price'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close Price'][i] = data['Close Price'][i]

# setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

# 分成 train and test
dataset = new_data.values

train = dataset[0:700,:]
test = dataset[700:,:]

# 构造 x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# 建立 LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

inputs = new_data[len(new_data) - len(test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

#for plotting
train = new_data[:700]
test = new_data[700:]
test['Predictions'] = closing_price
plt.plot(train['Close Price'])
plt.plot(test[['Close Price','Predictions']])
```

![](https://upload-images.jianshu.io/upload_images/1667471-f14e720747326ace.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


---

https://www.kaggle.com/dkmostafa/predicting-stock-market-using-linear-regression
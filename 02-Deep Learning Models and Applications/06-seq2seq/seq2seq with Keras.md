
seq2seq 的 keras 实现

上一篇 [seq2seq 入门](http://www.jianshu.com/p/1d3de928f40c) 提到了 cho 和 Sutskever 的两篇论文，今天来看一下如何用 keras 建立 seq2seq。

![](http://upload-images.jianshu.io/upload_images/1667471-22cf73f788c77217.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  

第一个 LSTM 为 Encoder，只在序列结束时输出一个语义向量，所以其 "return_sequences" 参数设置为 "False"

使用 "RepeatVector" 将 Encoder 的输出(最后一个 time step)复制 N 份作为 Decoder 的 N 次输入

第二个 LSTM 为 Decoder， 因为在每一个 time step 都输出，所以其 "return_sequences" 参数设置为 "True"

```
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, RepeatVector

def build_model(input_size, max_out_seq_len, hidden_size):
    
    model = Sequential()
    
    # Encoder(第一个 LSTM)     model.add( LSTM(input_dim=input_size, output_dim=hidden_size, return_sequences=False) )
    
    
    model.add( Dense(hidden_size, activation="relu") )
    
    # 使用 "RepeatVector" 将 Encoder 的输出(最后一个 time step)复制 N 份作为 Decoder 的 N 次输入
    model.add( RepeatVector(max_out_seq_len) )
    
    # Decoder(第二个 LSTM) 
    model.add( LSTM(hidden_size, return_sequences=True) )
    
    # TimeDistributed 是为了保证 Dense 和 Decoder 之间的一致
    model.add( TimeDistributed(Dense(output_dim=input_size, activation="linear")) )
    
    model.compile(loss="mse", optimizer='adam')

    return model
```

也可以用 GRU 作为 RNN 单元，代码如下，区别就是将 LSTM 处换成 GRU：

```
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, RepeatVector    

def build_model(input_size, seq_len, hidden_size):
    """建立一个 sequence to sequence 模型"""
    model = Sequential()
    model.add(GRU(input_dim=input_size, output_dim=hidden_size, return_sequences=False))
    model.add(Dense(hidden_size, activation="relu"))
    model.add(RepeatVector(seq_len))
    model.add(GRU(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(output_dim=input_size, activation="linear")))
    model.compile(loss="mse", optimizer='adam')

    return model
```
  
上面是一个最简单的 seq2seq 模型，因为没有将 Decoder 的每一个时刻的输出作为下一个时刻的输入。

---

当然，我们可以直接用 keras 的 seq2seq 模型：

https://github.com/farizrahman4u/seq2seq

下面是几个例子：

**简单的 seq2seq 模型：**

```
import seq2seq
from seq2seq.models import SimpleSeq2Seq

model = SimpleSeq2Seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=8)
model.compile(loss='mse', optimizer='rmsprop')
```

**深度 seq2seq 模型**：encoding 有 3 层,  decoding 有 3 层

```
import seq2seq
from seq2seq.models import SimpleSeq2Seq

model = SimpleSeq2Seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=8, depth=3)
model.compile(loss='mse', optimizer='rmsprop')
```

encoding 和 decoding 的层数也可以不同：encoding 有 4 层,  decoding 有 5 层

```
import seq2seq
from seq2seq.models import SimpleSeq2Seq

model = SimpleSeq2Seq(input_dim=5, hidden_dim=10, output_length=8, output_dim=20, depth=(4, 5))
model.compile(loss='mse', optimizer='rmsprop')
```

上面几种也是最简单的 SimpleSeq2Seq 的应用。

---

在论文 Sequence to Sequence Learning with Neural Networks 给出的 seq2seq 中，encoder 的隐藏层状态要传递给 decoder，而且 decoder 的每一个时刻的输出作为下一个时刻的输入，而且这里内置的模型中，还将隐藏层状态贯穿了整个 LSTM：

![](http://upload-images.jianshu.io/upload_images/1667471-db3b5b49858879f0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


```
import seq2seq
from seq2seq.models import Seq2Seq

model = Seq2Seq(batch_input_shape=(16, 7, 5), hidden_dim=10, output_length=8, output_dim=20, depth=4)
model.compile(loss='mse', optimizer='rmsprop')
```


cho 的这篇论文 Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation 中的 seq2seq 模型实现为：decoder 在每个时间点的语境向量都会获得一个 'peek'

![](http://upload-images.jianshu.io/upload_images/1667471-22cf73f788c77217.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  


```
import seq2seq
from seq2seq.models import Seq2Seq

model = Seq2Seq(batch_input_shape=(16, 7, 5), hidden_dim=10, output_length=8, output_dim=20, depth=4, peek=True)
model.compile(loss='mse', optimizer='rmsprop')
```

在论文 Neural Machine Translation by Jointly Learning to Align and Translate 中**带有注意力机制的 seq2seq**：没有隐藏状态的传播，而且 encoder 是双向的 LSTM

![](http://upload-images.jianshu.io/upload_images/1667471-85deb92d041a0687.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![](http://upload-images.jianshu.io/upload_images/1667471-0c01c5a67dfe32c1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


```
import seq2seq
from seq2seq.models import AttentionSeq2Seq

model = AttentionSeq2Seq(input_dim=5, input_length=7, hidden_dim=10, output_length=8, output_dim=20, depth=4)
model.compile(loss='mse', optimizer='rmsprop')
```

---

参考：
https://github.com/farizrahman4u/seq2seq
http://www.zmonster.me/2016/05/29/sequence_to_sequence_with_keras.html
http://jacoxu.com/encoder_decoder/

---
推荐阅读 [历史技术博文链接汇总](http://www.jianshu.com/p/28f02bb59fe5)
http://www.jianshu.com/p/28f02bb59fe5
也许可以找到你想要的
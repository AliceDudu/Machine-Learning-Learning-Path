Auto ML 一种自动完成机器学习任务的系统

Auto ML 是能够自动化完成一些机器学习任务的系统，

在 2018 年比较火，很多大公司都开源了各自的auto ml库，例如 Cloud AutoML, AUTO KERAS, Auto Sklearn, Auto Weka 等，

并被很多数据科学家预测在 2019 年仍然是机器学习的热点。



在做一个机器学习项目时，几乎每个环节都要人为地进行各种处理，各种尝试

例如数据预处理环节，一般就需要做这些步骤：

text vectorization

categorical data encoding (e.g., one hot)

missing values and outliers processing

rescaling (e.g., normalization, standardization, min-max scaling)

variables discretization

dimensionality reduction



还需要选择算法：

supervised or not, classification or regression, online or batch learning



特征工程，参数调节也是更复杂的部分，而且没有一个标准的模式可以遵循，随问题而变化



Auto ML 的目的就是要减少人为的操作，将特征工程，模型参数设置，算法选择部分由这个系统自动地去完成，并且要达到更好的性能，更快地运算



主要的算法有：

用于自动寻找最优神经网络结构的 NAS算法，

用于搜索超参的 贝叶斯算法，TPE模型等，

还有Google的 Bandit 算法，以及比较经典的遗传算法






以 Keras 为例：

在深度学习的库中，Keras 已经算是很简单明了的了，建立一个神经网络结构也比较方便，下面我们看看用 Keras 做 MNIST 任务的代码：



from __future__ import print_function

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K



batch_size = 128

num_classes = 10

epochs = 12



# input image dimensions

img_rows, img_cols = 28, 28



# the data, split between train and test sets

(x_train, y_train), (x_test, y_test) = mnist.load_data()



if K.image_data_format() == 'channels_first':

    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)

    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

    input_shape = (1, img_rows, img_cols)

else:

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')



# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                activation='relu',

                input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])



model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])





上面的代码中包含了下面这些步骤：

数据预处理，

设置模型参数，

建立模型，

训练模型，

评估模型



如果用 Auto-Keras 来做呢：



from keras.datasets import mnist

from autokeras.classifier import ImageClassifier



if __name__ == '__main__':

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape + (1,))

x_test = x_test.reshape(x_test.shape + (1,))



clf = ImageClassifier(verbose=True, augment=False)

clf.fit(x_train, y_train, time_limit=12 * 60 * 60)

clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)

y = clf.evaluate(x_test, y_test)

print(y * 100)



只需要 2 行，就自动化了前面的 数据预处理，设置模型参数



学习资源：

https://towardsdatascience.com/auto-keras-or-how-you-can-create-a-deep-learning-model-in-4-lines-of-code-b2ba448ccf5e




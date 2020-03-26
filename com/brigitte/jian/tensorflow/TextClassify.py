from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
# 将训练集按照 6:4 的比例进行切割，从而最终我们将得到 15,000
# 个训练样本, 10,000 个验证样本以及 25,000 个测试样本
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])
(train_data, validation_data), test_data = tfds.load(name="imdb_reviews",
                                                     split=(train_validation_split, tfds.Split.TEST),
                                                     as_supervised=True)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)
print(train_labels_batch)

# 首先创建一个使用 Tensorflow Hub 模型嵌入（embed）语句的Keras层
embedding = "https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])
# 第一层是 Tensorflow Hub 层。这一层使用一个预训练的保存好的模型来将句子映射为嵌入向量（embedding vector）。我们所使用的预训练文本嵌入
# （embedding）模型(google/tf2-preview/gnews-swivel-20dim/1)将句子切割为符号，嵌入（embed）每个符号然后进行合并。最终得到的维度是：
# (num_examples, embedding_dimension)。
# 该定长输出向量通过一个有 16 个隐层单元的全连接层（Dense）进行管道传输。
# 最后一层与单个输出结点紧密相连。使用 Sigmoid 激活函数，其函数值为介于 0 与 1 之间的浮点数，表示概率或置信水平。
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()
# 损失函数与优化器
# 一个模型需要损失函数和优化器来进行训练。由于这是一个二分类问题且模型输出概率值（一个使用 sigmoid 激活函数的单一单元层），我们将使用
# binary_crossentropy 损失函数。
#
# 这不是损失函数的唯一选择，例如，您可以选择 mean_squared_error 。但是，一般来说 binary_crossentropy 更适合处理概率——它能够度量概率分布之间
# 的“距离”，或者在我们的示例中，指的是度量 ground-truth 分布与预测值之间的“距离”。
#
# 稍后，当我们研究回归问题（例如，预测房价）时，我们将介绍如何使用另一种叫做均方误差的损失函数。
#
# 现在，配置模型来使用优化器和损失函数：
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 训练模型
# 以 512 个样本的 mini-batch 大小迭代 20 个 epoch 来训练模型。 这是指对 x_train 和 y_train 张量中所有样本的的 20 次迭代。在训练过程中，
# 监测来自验证集的 10,000 个样本上的损失值（loss）和准确率（accuracy）：
history = model.fit(train_data.shuffle(1000).batch(512), epochs=20, validation_data=validation_data.batch(512),
                    verbose=1)

# 评估模型
# 返回两个值。损失值（loss）（一个表示误差的数字，值越低越好）与准确率（accuracy）。
result = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, result):
    print("%s : %.3f" % (name, value))






















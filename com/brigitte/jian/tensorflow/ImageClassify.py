from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images.shape

print(train_images.shape)

print(test_images.shape)

print(train_labels)

print(test_labels)

plt.figure()

plt.imshow(train_images[1])

plt.colorbar()

plt.grid(False)

plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
# The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images from a
# two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels). Think of this layer
# as unstacking rows of pixels in the image and lining them up. This layer has no parameters to learn; it only reformats
# the data.
#
# After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers. These are
# densely connected, or fully connected, neural layers. The first Dense layer has 128 nodes (or neurons). The second
# (and last) layer returns a logits array with length of 10. Each node contains a score that indicates the current image
# belongs to one of the 10 classes.
model = keras.Sequential(
    [keras.layers.Flatten(input_shape=(28, 28)),
     keras.layers.Dense(128, activation='relu'),
     keras.layers.Dense(10)
     ]
)
# Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:
#
# Loss function —This measures how accurate the model is during training. You want to minimize this function to "steer"
# the model in the right direction.
# Optimizer —This is how the model is updated based on the data it sees and its loss function.
# Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the
# images that are correctly classified.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )
# Train the model
# Training the neural network model requires the following steps:
#
# Feed the training data to the model. In this examplecom/brigitte/jian/tensorflow/ImageClassify.py:39, the training data is
# in the train_images and train_labels
# arrays.
# The model learns to associate images and labels.
# You ask the model to make predictions about a test set—in this example, the test_images array.
# Verify that the predictions match the labels from the test_labels array.
# To start training, call the model.fit method—so called because it "fits" the model to the training data:
model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

predictions[0]
print(predictions[0])
np.argmax(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])


def plot_iamge(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_iamge(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_iamge(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

num_rows = 5
num_color = 3
num_images = num_rows * num_color
plt.figure(figsize=(2 * 2 * num_color, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_color, 2 * i + 1)
    plot_iamge(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_color, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

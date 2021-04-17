import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Keep cmd clean by removing debuggin information from tensorflow

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist # MNIST database

(x_train, y_train), (x_test, y_test) = mnist.load_data() # Sort data to images and labels

x_train = tf.keras.utils.normalize(x_train, axis=1) # Normalise the data, make it ready to be used by keras
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.load_model('main1_models\main1Model.model') # Loading model made by me earlier - for recognising numbers

predict = model.predict([x_test]) # Use the model to recognise numbers

# Shows results for first number - is 7
print(np.argmax(predict[0]))
plt.imshow(x_test[0])
plt.show()

#--------------------Creating Model----------------------------------

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=3)

# val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss, val_acc)

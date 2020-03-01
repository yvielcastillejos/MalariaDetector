#This program trains the machine using the np dataset

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Activation, Dense, Flatten, MaxPooling2D, Dropout

X = []
Y = []

X = np.load("features.npy")
Y = np.load("labels.npy")

# normalization
X = tf.keras.utils.normalize(X, axis=1)

# Using Sequential Function
model = tf.keras.models.Sequential()

# apply conv2 layer
model.add(Conv2D(64, (3, 3), input_shape = (70,70,3)))
model.add(Activation("relu"))# 70x70x3

# apply pooling
model.add(MaxPooling2D(pool_size=(2, 2)))

# apply another conv2 layer
model.add(Conv2D(64, (3, 3), activation=tf.nn.relu))
model.add(MaxPooling2D(pool_size=(2, 2)))

# apply dense layer
model.add(Flatten())
model.add(Dense(64))

# output dense layer
model.add(Dense(1, activation=tf.nn.sigmoid))

# train data
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, Y, epochs=2, batch_size=32, validation_split=0.3)

model.save("malariatester.model")

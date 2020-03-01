# input new data for testing.

import os
import cv2
import numpy
import tensorflow as tf

CATEGORIES = ["Parasitized","Uninfected"]


def prepare(DIR):
    pixel = 70
    img_array = cv2.imread(DIR)
    new_array = cv2.resize(img_array, (pixel, pixel))
    new_array = np.array(new_array).reshape(-1, pixel, pixel, 3)
    new_array = tf.keras.utils.normalization(new_array, axis=1)
    return new_array


new_model = tf.keras.models.load_model("malariatester.model")
Directory = ""
predictions = new_model.predict([prepare(Directory)])

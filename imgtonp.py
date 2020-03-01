#This code converts the images to arrays to be read by the CNN algorithm

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


Directory = "/Users/yvielcastillejos/Downloads/cell_images/cell_images"
CATEGORIES = ["Parasitized","Uninfected"] #0 is parasatized, 1 is Uninfected
X = []
Y =[]
data_temp = []

def converttomatrix(DIR):
    pixel = 70
    data_train = []
    for category in CATEGORIES:
        path = os.path.join(DIR, category)
        cnum = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
              img_array = cv2.imread(os.path.join(path,img))
              new_array = cv2.resize(img_array, (pixel,pixel))
              data_temp.append([new_array,cnum])
            #  plt.imshow(new_array)
            #  plt.show()
            except:
                pass
    random.shuffle(data_temp)
    for i in range(0,len(data_temp),1):
        X.append(data_temp[i][0])
        Y.append(data_temp[i][1])
    return X,Y

X,Y=converttomatrix(Directory)
X = np.array(X).reshape(-1, 70, 70, 3)
Y = np.array(Y)
np.save('features.npy',X)
np.save('labels.npy',Y)






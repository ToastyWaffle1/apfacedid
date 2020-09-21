import tensorflow as tf 
from tensorflow import keras 
import cv2 as cv
from cv2 import *
import skimage 
import numpy as np
from skimage import *
import shutil
print(tf.__version__)
#tf.compat.v1.enable_eager_execution
#tf.math.log
#tf.compat.v1.train.opimizer 
tf.nn.max_pool2d
train_images =[]
test_labels=[]
train_labels=[]
test_images=[]

class_names = ['African American', 'Asian', 'me', 'Hispanic']

for x in range(4):
    for i in range(1,2):
        train_images.append(skimage.transform.resize(io.imread("p\\"+class_names[x]+"\\"+class_names[x]+"("+str(i)+")"+".jpg"),[150,150,3]))
        train_labels.append(x)
train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)
            
train_labels = keras.utils.to_categorical(train_labels, 9)
print("AAAAAAAAA",train_labels.shape)
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(150,150,3)))
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
#model.add(keras.layers.MaxPooling2D((2, 2), data_format='channels_last'))
#model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))

#model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(9, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
from __future__ import absolute_import, division, print_function, unicode_literals

import yagmail

import matplotlib.pylab as plt

import tensorflow as tf
tf.compat.v1.enable_eager_execution
from tensorflow import keras

import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import layers
from skimage import io
from skimage import color
import sklearn.datasets as skd2
import skimage

import yagmail
import zipfile
zip_file = zipfile.ZipFile('second_model.zip', 'w')
zip_file.write('second_model.h5', compress_type=zipfile.ZIP_DEFLATED)
zip_file.close()
destination = input('')
body = input('')
filename = 'second_model.zip'
yag = yagmail.SMTP("owensaptest@gmail.com", "6Fitch66")
def emailres():
	yag.send(
		to=destination,
		subject="Aptest",
		contents=body,
		attachments=filename,
		)
#import kaggle_simpson_testset
		#model.load.(name.h5)
		#validation_data=(test_images, test_labels)
		#train_labels = keras.utils
	#model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
	#activation='relu'
	#input_shape=(32,32,3)))
#(train_images, train_labels), (test_images, test_labels)
#array = ['abraham_grampa_simpson_18.jpg', 'abraham_grampa_simpson_0.jpg', 'abraham_grampa_simpson_11.jpg', 'abraham_grampa_simpson_12.jpg', 'abraham_grampa_simpson_10.jpg', 'abraham_grampa_simpson_9.jpg', 'abraham_grampa_simpson_8.jpg', 'abraham_grampa_simpson_7.jpg', 'abraham_grampa_simpson_14.jpg', 'abraham_grampa_simpson_15.jpg']

train_images =[]
test_labels=[]
train_labels=[]
test_images=[]




class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
#(train_images, train_labels), (test_images, test_labels) =
for x in range(5):
		for i in range(1,500):
			train_images.append(skimage.transform.resize(io.imread("flowers//"+class_names[x]+"//"+class_names[x]+" ("+str(i)+")"+".jpg"),[150,150,3]))
			train_labels.append(x)
for x in range(5):
		for i in range(501,734):
			test_images.append(skimage.transform.resize(io.imread("flowers//"+class_names[x]+"//"+class_names[x]+" ("+str(i)+")"+".jpg"),[150,150,3]))
			test_labels.append(x)

train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels)
test_images = np.asarray(test_images)
train_labels = keras.utils.to_categorical(train_labels, 9)
test_labels = keras.utils.to_categorical(test_labels, 9)

print("AAAAAAAAA",test_labels.shape)
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

model.fit(train_images, train_labels,
          epochs=12,
          validation_data=(test_images, test_labels))
score = model.evaluate(test_images, test_labels)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.figure()
print(train_images)
print(train_images.shape)
plt.figure(figsize=(10,10))
for i in range(10):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid('off')
	plt.imshow(train_images[i], cmap=plt.cm.binary)
	#plt.xlabel(class_names[train_labels[i]])
plt.show()
model.save("second_model.h5")
emailres()

'''
plt.figure()
print(train_images)
print(train_images.shape)
plt.figure(figsize=(10,10))
for i in range(10):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid('off')
	plt.imshow(train_images[i], cmap=plt.cm.binary)
	#plt.xlabel(class_names[train_labels[i]])
plt.show()

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(10, activation=tf.nn.softmax)
	])
model.compile(optimizer='Adadelta',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
#plt.imshow(train_images)

img = io.imread('kaggle_simpson_testset/abraham_grampa_simpson_18.jpg', as_gray = True)
img2 = io.imread('kaggle_simpson_testset/abraham_grampa_simpson_0.jpg',as_gray = True)
plt.figure()
print(img)
plt.imshow(img)
plt.figure()
print(img2)
plt.imshow(img2)
plt.show()
print(train_images.shape)
#array = np.reshape(array,[28,28])
plt.figure(figsize=(10,10))
for i in range(10):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid('off')
	plt.imshow(train_images[i], cmap=plt.cm.binary)
	#plt.xlabel(class_names[train_labels[i]])
plt.show()

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(10, activation=tf.nn.softmax)
	])
model.compile(optimizer='Adadelta',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
'''

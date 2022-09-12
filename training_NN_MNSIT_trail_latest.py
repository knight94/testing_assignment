# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 13:53:45 2022

@author: nsahu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 21:55:41 2022

@author: nsahu
"""
"""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
Created on Thu Sep  1 15:44:05 2022

@author: naman
"""

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

print (tf.__version__) # 1.12.0

# tf.keras.backend.set_floatx('float16')
fashion_mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# returns 4 numpy arrays: 2 training sets and 2 test sets
# images: 28x28 arrays, pixel values: 0 to 255
# labels: array of integers: 0 to 9 => class of clothings
# Training set: 60,000 images, Testing set: 10,000 images

# class names are not included, need to create them to plot the images
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# print("train_images:", train_images.shape)
# print("test_images:", test_images.shape)

# # Visualize the first image from the training dataset
# plt.figure()
# plt.imshow(test_images[0])
# plt.colorbar()
# plt.grid(False)

levels = {(0, 25): 0, (26, 50): 10, (51, 75): 20, (76, 100): 30, (101,
125):40, (126, 150):50, (151, 175):60, (176, 200): 70, (201, 225): 80,
(226, 256): 90}
new_tr_img = np.zeros(train_images.shape).astype('int8')
for i in range(train_images.shape[0]):
     for x in range(28):
         for y in range(28):
             pix = train_images[i][x][y]
             for z in levels:
                 if pix >= z[0] and pix < z[1]:
                     new_tr_img[i][x][y] = levels[z]

new_test_img = np.zeros(test_images.shape).astype('int8')
for i in range(test_images.shape[0]):
     for x in range(28):
         for y in range(28):
             pix = test_images[i][x][y]
             for z in levels:
                 if pix >= z[0] and pix < z[1]:
                     new_test_img[i][x][y] = levels[z]

# scale the values to a range of 0 to 1 of both data sets
# train_images = train_images / 255.0
# test_images = test_images / 255.0
# new_tr_img = new_tr_img * 10
# new_test_img = new_test_img * 10

# display the first 25 images from the training set and
# display the class name below each image
# verify that data is in correct format
plt.figure(figsize=(10,10))
for i in range(25):
     plt.subplot(5,5, i+1)
     plt.xticks([])
     plt.yticks([])
     plt.grid(False)
     plt.imshow(new_tr_img[i], cmap=plt.cm.binary)
     plt.xlabel(class_names[train_labels[i]])

hidden_units = 64
# Step 1 - Build the architecture
# Model a simple 3-layer neural network
model_3 = keras.Sequential([
     keras.layers.Flatten(input_shape=(28,28)),
     keras.layers.Dense(hidden_units, activation=tf.nn.relu),
     keras.layers.Dense(10, activation=tf.nn.softmax)
])
model_3.summary()

# opt = keras.optimizers.Adam(learning_rate=0.1)
# Step 2 - Compile the model
model_3.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

#Step 3 - Train the model, by fitting it to the training data
# 5 epochs, and split the training set into 80/20 for validation
# model_3.fit(train_images, train_labels, epochs=5, validation_split=0.2)
model_3.fit(new_tr_img, train_labels, epochs=5, validation_split=0.2)


#Step 4 - Evaluate the model
test_loss, test_acc = model_3.evaluate(test_images, test_labels)
print("Model - 3 layers - test loss:", test_loss * 100)
print("Model - 3 layers - test accuracy:", test_acc * 100)

def relu(X):
    return np.maximum(0,X).astype('int16')

def softmax(X):
     expo = np.exp(X)
     expo_sum = np.sum(np.exp(X))
     return expo/expo_sum

def softmax_1(X):
    return X/np.sum(X)

w_ans = model_3.get_weights()
# ht, bins = np.histogram(ans[0].transpose()[0]*2)
# _ = plt.hist(ans[0].transpose()[0]*40)
w_scaled = list(w_ans)
for i in range(4):
    w_scaled[i] = (w_ans[i]*32).astype('int8')


ans = list(w_scaled)
# N = 100
count = 0
for i in range(10000):
    N = i
    test_input = new_test_img[N].ravel().reshape(1,784).astype('int16')
    
    hh_in = ((test_input.dot(ans[0]).transpose()).astype('int16') + ans[1].reshape(hidden_units,1)) >> 5
    hh_out = relu(hh_in.astype('int16'))
    
    out_in = ((hh_out.transpose().dot(ans[2]).transpose()).astype('int16') + ans[3].reshape(10,1)) >> 5
    # out_out = softmax_1(out_in)
    if (np.argmax(out_in) == test_labels[N]):
        count += 1

print(count/100)
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 23:13:05 2022

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

print (tf.__version__)

# tf.keras.backend.set_floatx('float16')
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

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

# levels = {(0, 25): 0, (26, 50): 10, (51, 75): 20, (76, 100): 30, (101,
# 125):40, (126, 150):50, (151, 175):60, (176, 200): 70, (201, 225): 80,
# (226, 256): 90}
# new_tr_img = np.zeros(train_images.shape)
# for i in range(train_images.shape[0]):
#      for x in range(28):
#          for y in range(28):
#              pix = train_images[i][x][y]
#              for z in levels:
#                  if pix >= z[0] and pix < z[1]:
#                      new_tr_img[i][x][y] = levels[z]

# new_test_img = np.zeros(test_images.shape)
# for i in range(test_images.shape[0]):
#      for x in range(28):
#          for y in range(28):
#              pix = test_images[i][x][y]
#              for z in levels:
#                  if pix >= z[0] and pix < z[1]:
#                      new_test_img[i][x][y] = levels[z]

# scale the values to a range of 0 to 1 of both data sets
# train_images = train_images / 255.0
# test_images = test_images / 255.0
# new_tr_img = new_tr_img / 9
# new_test_img = new_test_img/ 9

# display the first 25 images from the training set and
# display the class name below each image
# verify that data is in correct format
plt.figure(figsize=(10,10))
for i in range(25):
     plt.subplot(5,5, i+1)
     plt.xticks([])
     plt.yticks([])
     plt.grid(False)
     plt.imshow(train_images[i], cmap=plt.cm.binary)
     plt.xlabel(class_names[train_labels[i]])

# Step 1 - Build the architecture
# Model a simple 3-layer neural network
model_3 = keras.Sequential([
     keras.layers.Flatten(input_shape=(28,28)),
     keras.layers.Dense(128, activation=tf.nn.relu),
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
model_3.fit(train_images, train_labels, epochs=5, validation_split=0.2)


#Step 4 - Evaluate the model
test_loss, test_acc = model_3.evaluate(test_images, test_labels)
print("Model - 3 layers - test loss:", test_loss * 100)
print("Model - 3 layers - test accuracy:", test_acc * 100)

"""
Quantization of the model
"""
train_images_float32 = train_images.astype('float32')
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images_float32).batch(1).take(100):
    # Model has only one input so each data point has one element.
    yield [input_value]


converter = tf.lite.TFLiteConverter.from_keras_model(model_3)

tflite_model = converter.convert()

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model_quant = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)

import pathlib

tflite_models_dir = pathlib.Path("./mnist_tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the unquantized/float model:
tflite_model_file = tflite_models_dir/"mnist_model.tflite"
tflite_model_file.write_bytes(tflite_model)
# Save the quantized model:
tflite_model_quant_file = tflite_models_dir/"mnist_model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)


# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices):
  # global test_images

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices),), dtype=int)
  for i, test_image_index in enumerate(test_image_indices):
    test_image = test_images[test_image_index]
    test_label = test_labels[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    predictions[i] = output.argmax()

  return predictions

# Change this to test a different image
test_image_index = 1

## Helper function to test the models on one image
def test_model(tflite_file, test_image_index, model_type):
  global test_labels

  predictions = run_tflite_model(tflite_file, [test_image_index])

  plt.imshow(test_images[test_image_index])
  template = model_type + " Model \n True:{true}, Predicted:{predict}"
  _ = plt.title(template.format(true= str(test_labels[test_image_index]), predict=str(predictions[0])))
  plt.grid(False)

test_model(tflite_model_file, test_image_index, model_type="Float")

test_model(tflite_model_quant_file, test_image_index, model_type="Quantized")

# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file, model_type):
  global test_images
  global test_labels

  test_image_indices = range(test_images.shape[0])
  predictions = run_tflite_model(tflite_file, test_image_indices)

  accuracy = (np.sum(test_labels== predictions) * 100) / len(test_images)

  print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
      model_type, accuracy, len(test_images)))

evaluate_model(tflite_model_file, model_type="Float")

evaluate_model(tflite_model_quant_file, model_type="Quantized")


tf.lite.experimental.Analyzer.analyze(model_content=tflite_model_quant)

tf.lite.experimental.Analyzer.analyze(model_content=tflite_model)


interpreter = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
all_layers_details = interpreter.get_tensor_details()
for layers in all_layers_details:
    interpreter.get_tensor(layers['index'])

"""
Evaluating the qunatize model
"""
def relu(X):
    return np.maximum(0,X)

def softmax(X):
     expo = np.exp(X)
     expo_sum = np.sum(np.exp(X))
     return expo/expo_sum
 
def softmax_1(X):
     return X/np.sum(X)


# N = 100
count = 0
for i in range(10000):
     test_input = test_images[i].ravel().reshape(1,784).astype('int8')

     hh_in = ((np.dot(test_input, 0.004104394931346178 * weight_1.transpose()).transpose() + 0.004104394931346178 * bias_1.reshape(128,1))).astype('int8')
     hh_out = relu(hh_in)

     out_in = (np.dot(3.623868942260742 * (hh_out.transpose()+128), 0.002183682983741164 * weight_2.transpose()).transpose() + 0.007913380861282349 * bias_2.reshape(10,1)).astype('int8')
     out_out = softmax(1.4732587337493896*(out_in-5))
     if (np.argmax(out_out) == test_labels[i]):
         count += 1

print(count/100)

# def relu(X):
#     return np.maximum(0,X)

# def softmax(X):
#       expo = np.exp(X)
#       expo_sum = np.sum(np.exp(X))
#       return expo/expo_sum

# ans = model_3.get_weights()
# ans_float8 = list(ans)
# for i in range(4):
#       ans_float8[i] = ans[i].astype('int8')


# # N = 100
# count = 0
# for i in range(10000):
#       N = i
#       test_input = test_images[N].ravel().reshape(1,784).astype('uint8')

#       hh_in = np.dot(test_input, ans_float8[0]).transpose() + ans_float8[1].reshape(128,1)
#       hh_out = relu(hh_in)

#       out_in = np.dot(hh_out.transpose(), ans_float8[2]).transpose() + ans_float8[3].reshape(10,1)
#       out_out = softmax(out_in)
#       if (np.argmax(out_out) == test_labels[N]):
#           count += 1

# print(count/100)
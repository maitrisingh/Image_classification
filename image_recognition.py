#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:44:40 2020

@author: code
"""
import sys
import keras
import pandas
import matplotlib
import PIL
from keras.datasets import cifar10
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

print('Python: {}'.format(sys.version))
print('Keras: {}'.format(keras.__version__))

#import the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#lets determine the dataset characterstics
print('Training Images: {}'.format(X_train.shape))
print('Testing Images: {}'.format(X_test.shape))

#A single image
print(X_test[0].shape)

# create a grid of 3x3 image
for i in range(0,9):
    plt.subplot(330 + 1 + i)
    img = X_train[50 + i].transpose([0,1,2])
    plt.imshow(img)
    
#show the plot
plt.show()
#preprocessing the dataset

#fix random seed for reproducibility
seed = 6
np.random.seed(seed)

#load the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


#normalize the inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/ 255.0
X_test = X_test/ 255.0

print(X_train[0])


#class labels shape
print(y_train.shape)
print(y_train[0])

#hot encodes outputs
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)
num_class = Y_test.shape[1]

print(num_class)
print(Y_train.shape)
print(Y_train[0])



#Building the all CNN using the reference paper -striving for simplicity the all convolutional net
#using model C

# start by importing  necessary layers
from keras.models import Sequential
from keras.layers import Dropout, Activation, Conv2D, GlobalAveragePooling2D
from keras.optimizers import SGD

#define the model function

def allcnn(weights = None):
    
    #define model type- Sequential
    model = Sequential()
    
    
    #add model layers
    model.add(Conv2D(96, (3, 3), padding = 'same' ,input_shape=(32,32,3 )))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same', strides =(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same', strides =(2,2)))
    model.add(Dropout(0.5))
    
    
    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1), padding = 'valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1), padding = 'valid'))
    
    #add Global Average Pooling Layer with Softmax activation
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    #load the weights
    if weights:
       model.load_weights(weights)
        
    #return model
    return model


#define hyper parameters
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9

#define model
weights = 'all_cnn_weights_0.9088_0.4994.hdf5'
model = allcnn(weights)

#define the optimizer and compile model
sgd = SGD(lr = learning_rate, decay = weight_decay, momentum = momentum, nesterov= True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics =['accuracy'])

#print model summary
print (model.summary())      


#train the model with pretrainned weights
scores = model.evaluate(X_test, Y_test, verbose = 1)
print('Accuracy: {}'.format(scores[1]))

      
      
    
    
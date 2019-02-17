#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:28:40 2018

@author: KaranJaisingh
"""
# import required libraries
import numpy as np
import pandas as pd
from imutils import paths

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


# override truncated images error
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# define constants and parameters
train_data_dir = "train"
test_data_dir = "test"
TRAIN = len(list(paths.list_images(train_data_dir)))
TEST = len(list(paths.list_images(test_data_dir)))
BS = 8
EPOCHS = 20
img_width, img_height = 300, 300


# create CNN model outline
classifier = Sequential()

classifier.add(Conv2D(128, (2,2), 
                      input_shape = (img_width, img_height, 3), 
                      activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(64, (3,3), activation = 'relu'))
classifier.add(Conv2D(64, (2,2), activation = 'relu'))
classifier.add(Conv2D(64, (1,1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (2,2), activation = 'relu'))
classifier.add(Conv2D(32, (2,2), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units = 10, activation = 'relu'))
classifier.add(Dropout(0.3))
classifier.add(Dense(units = 1, activation = 'sigmoid'))


# compile model
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])


# create data generators and data pathways
from keras.preprocessing.image import ImageDataGenerator

trainAug = ImageDataGenerator(rescale = 1./255,
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               horizontal_flip = True,
                               fill_mode = "nearest")

testAug = ImageDataGenerator(rescale = 1./255,
                             fill_mode = "nearest")

trainGen = trainAug.flow_from_directory('train',
                                         target_size = (img_width, img_height),
                                         batch_size = BS,
                                         class_mode = 'binary')

testGen = testAug.flow_from_directory('test',
                                    target_size = (img_width, img_height),
                                    batch_size = BS,
                                    class_mode = 'binary')

classes = trainGen.class_indices    
print(classes)


# train model
H = classifier.fit_generator(trainGen,
                             epochs = EPOCHS,
                             steps_per_epoch = TRAIN // BS)


# print confirmation and save model
print("CNN Trained")
classifier.save('model.h5')
print("CNN Saved")
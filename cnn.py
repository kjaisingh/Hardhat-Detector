#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:28:40 2018

@author: KaranJaisingh
"""
# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imutils import paths

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.applications import VGG19
from keras.models import Model 
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# override truncated images error
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# define constants and parameters
train_data_dir = "train"
test_data_dir = "test"
TRAIN = len(list(paths.list_images(train_data_dir)))
TEST = len(list(paths.list_images(test_data_dir)))
BS = 10
EPOCHS = 20
img_width, img_height = 300, 300


# create model skeleton
base_model = VGG19(weights = "imagenet", include_top=False, 
                   input_shape = (img_width, img_height, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation = "relu")(x)
x = Dropout(0.4)(x)
x = Dense(256, activation = "relu")(x)
x = Dropout(0.1)(x)
preds = Dense(2, activation = "softmax")(x)

model = Model(input = base_model.input, output = preds)

for i,layer in enumerate(model.layers):
    print(i,layer.name)

for layer in model.layers[:14]:
    layer.trainable=False
for layer in model.layers[14:]:
    layer.trainable=True
    
model.summary()


# compile model
early = EarlyStopping(monitor = 'val_acc', min_delta = 0, 
                      patience = 10, verbose= 1 , mode = 'auto')

model.compile(loss = "categorical_crossentropy", 
              optimizer = SGD(lr=0.001, momentum=0.9), 
              metrics=["accuracy"])



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
                                         shuffle = True,
                                         class_mode = 'categorical')

testGen = testAug.flow_from_directory('test',
                                    target_size = (img_width, img_height),
                                    batch_size = BS,
                                    class_mode = 'categorical')


# train model
classes = trainGen.class_indices    
print(classes)

H = model.fit_generator(trainGen, epochs = EPOCHS,
                        steps_per_epoch = TRAIN // BS)
print("CNN Trained")


# save model
model.save('model.h5')
print("CNN Saved")


# plotting training data
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.jpg")


# generating predictions using model
testGen.reset()
predictions = model.predict_generator(testGen, steps = (TEST // BS) + 1) 
predictions = np.argmax(predictions, axis=1)


# evaluating predictions
acc = str(round(accuracy_score(testGen.classes, predictions, normalize=True) * 100, 2))
print("Test set accuracy: " + acc + "%") 
print("Classification report: \n" + classification_report(testGen.classes, predictions,
                            target_names=testGen.class_indices.keys())) 
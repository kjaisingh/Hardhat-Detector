#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:46:58 2018

@author: KaranJaisingh
"""
# import required libraries
import cv2
import argparse
import os
import numpy as np

from keras.models import load_model
from keras.optimizers import SGD


# define constants and parameters
img_width, img_height = 300, 300


# create argument parser for custom image input
parser = argparse.ArgumentParser(description = 'This is a Hardhat Detection program')
parser.add_argument("-i","--image", type = str, 
                    help = "File name of image to classify", 
                    default = "test-neg.jpg")
fileName = parser.parse_args().image


# load model
model = load_model('model.h5')
model.compile(loss = "categorical_crossentropy", 
              optimizer = SGD(lr=0.001, momentum=0.9), 
              metrics=["accuracy"])


# make prediction if image specified exists
if(os.path.isfile(fileName)):
    img = cv2.imread(fileName)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype("float") /  255.0
    img = np.reshape(img, [1, img_width, img_height, 3])

    result = model.predict(img)
    pred = np.argmax(result, axis=1)

    if(pred[0] == 0):
        print("No hardhat is being worn.")
    else:
        print("A hardhat is being worn.")
    

# return false statement if image specified does not exist
else:
    print("The directory " + fileName + " could not be located.")
    
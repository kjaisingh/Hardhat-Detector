#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:46:58 2018

@author: KaranJaisingh
"""
# import required libraries
from keras.models import load_model
import cv2
import argparse
import os
import numpy as np


# define constants and parameters
img_width, img_height = 300, 300


# create argument parser for custom image input
parser = argparse.ArgumentParser(description = 'This is a Hardhat Detection program')
parser.add_argument("-i","--image", type = str, 
                    help = "File name of image to classify", 
                    default = "test_pos.jpg")
fileName = parser.parse_args().image


# load model
classifier = load_model('model.h5')
classifier.compile(optimizer = 'adam', 
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

# make prediction if image specified exists
if(os.path.isfile(fileName)):
    img = cv2.imread(fileName)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype("float") /  255.0
    img = np.reshape(img, [1, img_width, img_height, 3])

    pred = classifier.predict(img)
    
    predictedClass = "UNRECOGNIZABLE"
    if(pred <= 0.5):
        predictedClass = "No hardhat"
    else:
        predictedClass = "Hardhat"
    
    print("The predicted class is: ", predictedClass)
    print("The model's predicted score is: ", pred[0][0])

# return false statement if image specified does not exist
else:
    print("The directory " + fileName + " could not be located.")
    
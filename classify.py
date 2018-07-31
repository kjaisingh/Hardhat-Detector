#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 14:46:58 2018

@author: KaranJaisingh
"""

from keras import backend as K
from keras.models import load_model

import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='This is a Hardhat Detection program')
parser.add_argument("-i","--image", type=str, help="File name of image to classify", default="test_pos.jpg")
fileName = parser.parse_args().image

classifier = load_model('detection_model.h5')

img = cv2.imread(fileName)
img = cv2.resize(img, (300, 300))

predictedClass = classifier.predict_classes(imgData)
print('The predicted class for the input image is: ', predictedClass)
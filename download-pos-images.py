#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 08:30:56 2018

@author: KaranJaisingh
"""

import cv2
from six.moves import urllib
import numpy as np
import os

def store_raw_images():
	pos_image_link = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03492922'
	pos_image_urls = urllib.request.urlopen(pos_image_link).read().decode()

	if not os.path.exists('pos'):
		os.makedirs('pos')

	pic_num = 1

	for i in pos_image_urls.split('\n'):
		try:
			print(pic_num, i)
			urllib.request.urlretrieve(i, 'pos/'+str(pic_num)+'.jpg')
			img = cv2.imread('pos/'+str(pic_num)+'.jpg')
			resized_image = cv2.resize(img, (300, 300))
			cv2.imwrite('pos/'+str(pic_num)+'.jpg', resized_image)
			pic_num += 1
			if(pic_num > 1000):
				return

		except Exception as e:
			print(str(e))

store_raw_images()

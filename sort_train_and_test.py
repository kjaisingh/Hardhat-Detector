#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 18:37:47 2018

@author: KaranJaisingh
"""

import os
import shutil

neg_files = os.listdir("neg")
pos_files = os.listdir("pos")

neg = len(neg_files)
pos = len(pos_files)

neg_train = int(neg * 0.8)
neg_test = neg - neg_train

pos_train = int(pos * 0.8)
pos_test = pos - pos_train

os.makedirs('train/neg')
os.makedirs('train/pos')
os.makedirs('test/neg')
os.makedirs('test/pos')

for i in range(0, neg_train):
    file = "neg/" + neg_files[i]
    shutil.move(file, "train/neg")
    
for i in range(neg_train, neg):
    file = "neg/" + neg_files[i]
    shutil.move(file, "test/neg")
    
for i in range(0, pos_train):
    file = "pos/" + pos_files[i]
    shutil.move(file, "train/pos")
    
for i in range(pos_train, pos):
    file = "pos/" + pos_files[i]
    shutil.move(file, "test/pos")
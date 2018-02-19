#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 08:38:36 2018

@author: KaranJaisingh
"""
import os
import numpy as np

def create_pos_n_neg():
	for file_type in ['neg']:

		for img in os.listdir(file_type):
			if file_type == 'neg':
				line = file_type+'/'+img+'\n'
				with open('bg.txt', 'a') as f:
					f.write(line)
			elif file_type == 'pos':
                		line = file_type+'/'+img+' 1 0 0 100 100\n'
                		with open('info.dat','a') as f:
                    			f.write(line)

create_pos_n_neg()

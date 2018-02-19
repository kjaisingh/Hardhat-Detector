# hardhat-detector
A convolutional neural network implementation of a script that detects whether an individual is wearing a hardhat or not. 


Intended implementation is for a construction company wishing to have a detection system that scans a construction worker walking, and checks that they are wearing the necessary safety equipment, preventing the need of an individual to manually do so and increasing workspace safety.



This repository contains five different Python files, and should be run in the following order from the root directory of the project. The command to run each file is 'python <file_name.py>':
1. download-neg-images.py : downloads negative (false) images from the ImageNet dataset and stores it in a directory locally
2. download-pos-images.py : downloads positive (true) images from the ImageNet dataset and stores it in a directory locally
3. store-images.py : creates a text file that holds the information for all images downloaded
4. sort_train_and_test.py : splits the downloaded images into training and test set, and sorts them into directories suitable for use with Keras functions
5. download-neg-images-py : creates and trains the convolutional neural network

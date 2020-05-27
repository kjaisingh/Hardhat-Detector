# Hardhat Detector
### Update in May 2020: Unfortunately, the ImageNet server will be down for an indefinite period of time, so the training image data generation process will not work until the server is back up.
A Convolutional Neural Network implementation of a script that detects whether an individual is wearing a hardhat or not.


The intended implementation for this system is for a construction company wishing to have a detection system that scans a construction worker walking, and checks that they are wearing the necessary safety equipment, preventing the need of an individual to manually do so and increasing work environment safety.


This repository contains numerous files that allow for the downloading of training and testing images, the sorting of these images, the construction of a model and the implementation of a model. All commands must be run in the working directory of the folder, and can be run as follows:
1. Download negative (false) images from the ImageNet dataset and stores it in a directory locally.
~~~~
python download-neg-images.py
~~~~~~~~ 

2. Download positive (true) images from the ImageNet dataset and stores it in a directory locally.
~~~~
python download-pos-images.py
~~~~~~~~ 

3. Certain downloaded images may simply be empty or irrepresentative images - to improve the accuracy of the algorithm, delete these manually.

4. Split the downloaded images into training and test set, and sorts them into directories suitable for use with Keras functions.
~~~~
python sort_train_and_test.py
~~~~~~~~ 

5. Create and train the convolutional neural network.
~~~~
python cnn.py
~~~~~~~~ 

6. Pass in images to be scanned for the presence of hardhats using the parser argument -i, with 'imageFileName' being a placeholder for the file name of the image. The default is 'test_pos.jpg', also part of the repository. The classification result is printed in the console.
~~~~
python classify.py -i <imageFileName>
~~~~~~~~ 

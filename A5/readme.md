# Localization Problem

## Problem Statement

### Task 1 

For the given datasets we are to do the following tasks:

- Localize the objects using regression and give class labels to them.
- For this task, we are provided with three class data set which includes Knuckles, Vein, and Palm. 
There are three folders viz; Knuckles, Palm, and Vein, in which data folder along with groundtruth.txt file is given. 
In groundtruth.txt file, each line containing five things: Image Name/Image Path, x1,y1 (left top most point of box) and x2,y2 ( right bottom up point of box), and Class label.

### Task 2 

For the given data-set comprising of four-slap finger prints, our task is to localize four objects instead of one. 
We are provided with Fourslap fingerprint images and corresponding ground truth. 
In ground truth folder, for each image we have one text file in which four lines are given (Each line containing 4 values (x1,y1,x2,y2)). 
We design a Neural Network that can localize the four slab fingerprint given image as input. 
Augmentor tool used for data augmentation.

We also wrote script to save IOU for each image in a text file line by line.

## (Notes about code)

### Task 1  :

Since the dataset consists of 3 different classes with images of varying sizes, we have used resizeImages.py to bring them all to the size of (459,352). Apart from that, knuckles ground truth required replacing space with underscore so we used rename_paths.py for this. Thus the same things need to be applied to the test dataset. 

## Details about architecture

### Task 1  :

The network used consists of a pretrained VGG16 network (only till block_5 of convolutional layers), followed by our own convolutional layers. We are not using any activation on regression head,softmax on classification head, and loss for classification head is categorical cross entropy and for regression head is MSE.

### Task 2  :

The network is similar to that used in task 1, except there's a single regression head with 16 neurons in last layer.

### Prerequisites

- Python 3.6
- Tensorflow 1.13.1
- Matplotlib: 3.0.2
- Jupyter 5.7.4 (optional)
- Numpy 1.16.1
- Pandas 0.24.1

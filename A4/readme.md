## Foundations of Convolutional Neural Networks

Folder has code for a CNN model to classify the images of the line dataset [[tar file]](https://github.com/moha23/DL/tree/master/A1/data/lines) into the respective 96 classes and MNIST dataset into 10 classes.

- Part 1: A network implementation with the following architecture:
  1. 7x7 Convolutional Layer with 32 filters and stride of 1. 
  2. ReLU Activation Layer.
  3. Batch Normalization Layer
  4. 2x2 Max Pooling layer with a stride of 2
  5. fully connected layer with 1024 output units.
  6. ReLU Activation Layer.
- Part 2: Network architecture to achieve greater accuracy

Prerequisites:
- Python 3.6
- Tensorflow 1.13.1
- Numpy 1.16.1

Inbuilt libraries of Python:
- Pathlib
- OS
- Functools
- random

References:
Tensorflow documentation on Estimators.

Part_1
As per suggested in the question, network architecture was tried with the 7 specifications.
In addition,  different filters sizes, pooling dimensions apart from the ones suggested were also tried.

Part_2:
Classification was tried on Lines dataset and 98% acuracy is obtained consistently. 

High level API called Estimators is used to execute the train and eval functions.

General Inferences:
Optimum results were achieved when the following hyper parameters were used - 3 convolution layers, Adam optimiser with epsilon - 0.1, Batch Normalization .
Using dropouts did not help much.
Max pooling of different sizes were tried - not much difference in the results.
In any case, with increase in epochs, accuracy is increasing invariably.

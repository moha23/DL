# Multi-Head Classification

## Getting Started

This folder has code for a non-sequential network which has architecture as seen in the following diagram :

![multihead](https://i.imgur.com/uwiVKGil.png)

The non-sequential network is for classifying the line dataset. Line dataset was created in a different assignment and is uploaded in repository [A1/data/lines](https://github.com/moha23/DL/tree/master/A1/data/lines)

This network has 4 outputs based on the 4 kind of variations(length, width, color, angle) of lines. The network architecture is divided into two parts a) Feature network and b) Classification heads. The feature network is responsible for extracting the required features from the input and attached to it are the four classification heads one for each variation.

The first 3 classification heads are for 2 class problems namely length, width and color classification. In all these the final layer contains a single neuron with a sigmoid activation followed by binary crossentropy loss.

The last classification head is a 12 class problem for each 12 angles of variation. In this the final layer contains 12 neurons with softmax activation and Categorical Cross entropy loss.

In this repository folder, multihead.py(python) and multihead.ipynb(jupyter) both have the code for training and testing of the line dataset. The jupyter code includes some results and confusion matrices for viewing. The saved-model folder contains the saved model of the trained network (jupyter) which can be restored using tf.train.saver.restore. The extras folder contains some code for reading line dataset into numpy arrays, and a simple convolutional network for classifying the line dataset. A report detailing results of experiments and inferences is also included.


### Prerequisites

- Python 3.6
- Tensorflow 1.13.1
- Matplotlib: 3.0.2
- Jupyter 5.7.4 (optional)
- Numpy 1.16.1
- Pandas 0.24.1

### Reference

- [Hvass Labs](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb)





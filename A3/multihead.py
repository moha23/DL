import numpy as np
import pandas as pd
import cv2
import glob
import os

x=np.empty([1000,28,28,3])
y=np.empty([1000,96])
g=np.empty([1000,96])
y_int=np.empty([1000])
g_int=np.empty([1000])

#one_hot_encoded
y_len=np.empty([1000,2])
y_wid=np.empty([1000,2])
y_col=np.empty([1000,2])
y_ang=np.empty([1000,12])

g_len=np.empty([1000,2])
g_wid=np.empty([1000,2])
g_col=np.empty([1000,2])
g_ang=np.empty([1000,12])

#true class
y_len_int=np.empty([1000,1])
y_wid_int=np.empty([1000,1])
y_col_int=np.empty([1000,1])
y_ang_int=np.empty([1000])

g_len_int=np.empty([1000,1])
g_wid_int=np.empty([1000,1])
g_col_int=np.empty([1000,1])
g_ang_int=np.empty([1000])

first = 1
num=0
for root, dirs, files in os.walk("pathname"):
    os.chdir(root)
    images = np.array([cv2.imread(file) for file in glob.glob(root+"/*.jpg")])
    num_of_images=images.shape[0]

    if num_of_images == 0: 
        continue
    else:
        if first == 1:
            x = images
            orgname = os.path.basename(os.path.normpath(root))
            y_len.fill(0)
            y_wid.fill(0)
            y_col.fill(0)
            y_ang.fill(0)
            name=orgname.split('_', 4)
            length=int(name[0])
            width=int(name[1])
            colour=int(name[3])
            angle=int(name[2])
            #print(colour)
            for i in range(0,999):
                y_len[i,length]=1
                y_wid[i,width]=1
                y_col[i,colour]=1
                y_ang[i,angle]=1
                
            y_len_int.fill(length)
            y_wid_int.fill(width)
            y_col_int.fill(colour)
            y_ang_int.fill(angle)
            
            d={'class':num,'label':[orgname],'length':[length],'width':[width],'colour':[colour],'angle':[angle]}
            corr = pd.DataFrame(data = d)
            first = 0
            num=num+1

        else:
            x = np.concatenate((x,images),axis=0)
            orgname = os.path.basename(os.path.normpath(root))
            name=orgname.split('_', 4)
            g_len.fill(0)
            g_wid.fill(0)
            g_col.fill(0)
            g_ang.fill(0)
            length=int(name[0])
            width=int(name[1])
            colour=int(name[3])
            angle=int(name[2])
            #print(colour)
            for i in range(0,999):
                g_len[i,length]=1
                g_wid[i,width]=1
                g_col[i,colour]=1
                g_ang[i,angle]=1
            y_len=np.concatenate((y_len,g_len),axis=0)
            y_wid =np.concatenate((y_wid,g_wid),axis=0)
            y_col =np.concatenate((y_col,g_col),axis=0)
            y_ang=np.concatenate((y_ang,g_ang),axis=0)
            
            g_len_int.fill(length)
            g_wid_int.fill(width)
            g_col_int.fill(colour)
            g_ang_int.fill(angle)
            
            y_len_int=np.concatenate((y_len_int,g_len_int),axis=0)
            y_wid_int=np.concatenate((y_wid_int,g_wid_int),axis=0)
            y_col_int=np.concatenate((y_col_int,g_col_int),axis=0)
            y_ang_int=np.concatenate((y_ang_int,g_ang_int),axis=0)
            
            d={'class':num,'label':[orgname],'length':[length],'width':[width],'colour':[colour],'angle':[angle]}
            newcorr = pd.DataFrame(data=d)
            corr = corr.append(newcorr)
            num=num+1
#print(corr)
x=np.reshape(x,[96000,2352])
from sklearn.model_selection import train_test_split
x_train, x_test, y_len_train, y_len_test,y_wid_train, y_wid_test,y_col_train, y_col_test,y_ang_train, y_ang_test,y_len_int_train, y_len_int_test,y_wid_int_train, y_wid_int_test,y_col_int_train, y_col_int_test,y_ang_int_train, y_ang_int_test = train_test_split(x, y_len,y_wid,y_col,y_ang, y_len_int,y_wid_int,y_col_int,y_ang_int, test_size=0.25, random_state=42)



def random_batch(x,y_len,y_wid,y_col,y_ang,y_len_int,y_wid_int,y_col_int,y_ang_int,batch_size):
        
        num_train = 72000
        # Create a random index into the training-set.
        idx = np.random.randint(low=0, high=num_train, size=batch_size)

        # Use the index to lookup random training-data.
        x_batch = x[idx]
        y_len_batch = y_len[idx]
        y_len_int_batch = y_len_int[idx]
        y_wid_batch = y_wid[idx]
        y_wid_int_batch = y_wid_int[idx]
        y_col_batch = y_col[idx]
        y_col_int_batch = y_col_int[idx]
        y_ang_batch = y_ang[idx]
        y_ang_int_batch = y_ang_int[idx]

        return x_batch, y_len_int_batch, y_wid_int_batch,y_col_int_batch,y_ang_batch

def isGreater(x,y):
    res = tf.math.greater(x,y)
    return res

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math



# The number of pixels in each dimension of an image.
img_size = 28

# The images are stored in one-dimensional arrays of this length.
img_size_flat = 2352

# Tuple with height and width of images used to reshape arrays.
img_shape = [28,28,3]

# Number of classes
#num_classes = 96

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 3



def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


#feature extractor with two convolutional layers
with tf.variable_scope('FeaureExtractor', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('ConvNetLayer1', reuse=tf.AUTO_REUSE):
        x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
        x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
        weights1 = new_weights(shape=[5,5,3,16])
        biases1 = new_biases(length=16)
        conv1 = tf.nn.conv2d(input=x_image,
                             filter=weights1,
                             strides=[1, 1, 1, 1],
                             padding='SAME',
                             name = 'conv1')
        conv1 += biases1
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(value=conv1,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='maxpooledconv1')
    with tf.variable_scope('ConvNetLayer2', reuse=tf.AUTO_REUSE):
        weights2 = new_weights(shape=[5,5,16,36])
        biases2 = new_biases(length=36)
        conv2 = tf.nn.conv2d(input=conv1,
                             filter=weights2,
                             strides=[1, 1, 1, 1],
                             padding='SAME',
                             name = 'conv2')
        conv2 += biases2
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(value=conv2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='maxpooledconv2')
    with tf.variable_scope('FlattenLayer',reuse=tf.AUTO_REUSE):
        layer_flat = tf.contrib.layers.flatten(conv2)
        num_features = layer_flat.shape[1]
        print(num_features)


y_len_true = tf.placeholder(tf.float32, shape=[None, 1], name='y_len_true')
y_wid_true = tf.placeholder(tf.float32, shape=[None, 1], name='y_wid_true')
y_col_true = tf.placeholder(tf.float32, shape=[None, 1], name='y_col_true')
y_ang_true = tf.placeholder(tf.float32, shape=[None, 12], name='y_ang_true')



y_len_true_cls =y_len_true
y_wid_true_cls =y_wid_true
y_col_true_cls =y_col_true
y_ang_true_cls = tf.argmax(y_ang_true, axis=1)



#Length head with two fully connected layers
with tf.variable_scope('LengthHead', reuse=tf.AUTO_REUSE):
    layer_len_fc1 = tf.layers.dense(layer_flat,128,activation='relu',name='LengthFC1',reuse=tf.AUTO_REUSE)
    layer_len_fc2 = tf.layers.dense(layer_len_fc1,1,name='LengthFC2',reuse=tf.AUTO_REUSE)
    y_len_pred = tf.nn.sigmoid(layer_len_fc2)
    if(isGreater(y_len_pred,0.5) is True):
        y_len_pred_cls = tf.math.ceil(y_len_pred)
    else:
        y_len_pred_cls = tf.math.floor(y_len_pred)
    cross_entropy_len = tf.nn.sigmoid_cross_entropy_with_logits(logits=layer_len_fc2,
                                                        labels=y_len_true)
    cost_len = tf.reduce_mean(cross_entropy_len)
    correct_prediction_len = tf.equal(y_len_pred_cls, y_len_true_cls)
    accuracy_len = tf.reduce_mean(tf.cast(correct_prediction_len, tf.float32))
    
#Width head with two fully connected layers
with tf.variable_scope('WidthHead', reuse=tf.AUTO_REUSE):
    layer_wid_fc1 = tf.layers.dense(layer_flat,128,activation='relu',name='WidthFC1',reuse=tf.AUTO_REUSE)
    layer_wid_fc2 = tf.layers.dense(layer_wid_fc1,1,name='WidthFC2',reuse = tf.AUTO_REUSE)
    y_wid_pred = tf.nn.sigmoid(layer_wid_fc2)
    if(isGreater(y_wid_pred,0.5) is True):
        y_wid_pred_cls = tf.math.ceil(y_wid_pred)
    else:
        y_wid_pred_cls = tf.math.floor(y_wid_pred)
    cross_entropy_wid = tf.nn.sigmoid_cross_entropy_with_logits(logits=layer_wid_fc2,
                                                        labels=y_wid_true)
    cost_wid = tf.reduce_mean(cross_entropy_wid)
    correct_prediction_wid = tf.equal(y_wid_pred_cls, y_wid_true_cls)
    accuracy_wid = tf.reduce_mean(tf.cast(correct_prediction_wid, tf.float32))
  
#Colour head with two fully connected layers  
with tf.variable_scope('ColourHead', reuse=tf.AUTO_REUSE):
    layer_col_fc1 = tf.layers.dense(layer_flat,128,activation='relu',name='ColourFC1',reuse=tf.AUTO_REUSE)
    layer_col_fc2 = tf.layers.dense(layer_col_fc1,1,name='ColourFC2',reuse=tf.AUTO_REUSE)
    y_col_pred = tf.nn.sigmoid(layer_col_fc2)
    if(isGreater(y_col_pred,0.5) is True):
        y_col_pred_cls = tf.math.ceil(y_col_pred)
    else:
        y_col_pred_cls = tf.math.floor(y_col_pred)
    cross_entropy_col = tf.nn.sigmoid_cross_entropy_with_logits(logits=layer_col_fc2,
                                                        labels=y_col_true)
    cost_col = tf.reduce_mean(cross_entropy_col)
    correct_prediction_col = tf.equal(y_col_pred_cls, y_col_true_cls)
    accuracy_col = tf.reduce_mean(tf.cast(correct_prediction_col, tf.float32))

#Angle head with two fully connected layers    
with tf.variable_scope('AngleHead', reuse=tf.AUTO_REUSE):
    layer_ang_fc1 = tf.layers.dense(layer_flat,128,activation='relu',name='AngleFC1',reuse=tf.AUTO_REUSE)
    layer_ang_fc2 = tf.layers.dense(layer_ang_fc1,12,name='AngleFC2',reuse=tf.AUTO_REUSE)
    y_ang_pred = tf.nn.softmax(layer_ang_fc2)
    y_ang_pred_cls = tf.argmax(y_ang_pred, axis=1)
    cross_entropy_ang = tf.nn.softmax_cross_entropy_with_logits(logits=layer_ang_fc2,
                                                        labels=y_ang_true)
    cost_ang = tf.reduce_mean(cross_entropy_ang)
    correct_prediction_ang = tf.equal(y_ang_pred_cls, y_ang_true_cls)
    accuracy_ang = tf.reduce_mean(tf.cast(correct_prediction_ang, tf.float32))
    
total_cost = cost_len + cost_wid +cost_col+cost_ang
optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(total_cost)
session = tf.Session()
session.run(tf.global_variables_initializer())



train_batch_size = 64

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    loss_list=[]
    acc_list=[]
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        x_batch,y_len_batch,y_wid_batch,y_col_batch,y_ang_batch = random_batch(x_train,y_len_train,y_wid_train,y_col_train,y_ang_train,y_len_int_train,y_wid_int_train,y_col_int_train,y_ang_int_train,batch_size=train_batch_size)
    
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_len_true: y_len_batch,
                           y_wid_true: y_wid_batch,
                           y_col_true: y_col_batch,
                           y_ang_true: y_ang_batch
                        }

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc_len = session.run(accuracy_len, feed_dict=feed_dict_train)
            acc_wid = session.run(accuracy_wid, feed_dict=feed_dict_train)
            acc_col = session.run(accuracy_col, feed_dict=feed_dict_train)
            acc_ang = session.run(accuracy_ang, feed_dict=feed_dict_train)
            loss = session.run(total_cost, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy length: {1:>6.1%} width: {2:>6.1%} colour: {3:>6.1%} angle: {4:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc_len,acc_wid,acc_col,acc_ang))
            
            #plot loss
            loss_list.append(loss)
            plt.plot(loss_list)
            
    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy():

    # Number of images in the test-set.
    num_test = 24000

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred_len = np.zeros(shape=[num_test,1], dtype=np.int)
    cls_pred_wid = np.zeros(shape=[num_test,1], dtype=np.int)
    cls_pred_col = np.zeros(shape=[num_test,1], dtype=np.int)
    cls_pred_ang = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = x_test[i:j, :]

        # Get the associated labels.
        labels_len = y_len_int_test[i:j,:]
        labels_wid = y_wid_int_test[i:j,:]
        labels_col = y_col_int_test[i:j,:]
        labels_ang = y_ang_test[i:j,:]
        

        # Create a feed-dict with these images and labels.
        feed_dict_len = {x: images,
                     y_len_true: labels_len}
        feed_dict_wid = {x: images,
                     y_wid_true: labels_wid}
        feed_dict_col = {x: images,
                     y_col_true: labels_col}
        feed_dict_ang = {x: images,
                     y_ang_true: labels_ang}

        # Calculate the predicted class using TensorFlow.
        cls_pred_len[i:j,:] = session.run(y_len_pred_cls, feed_dict=feed_dict_len)
        cls_pred_wid[i:j,:] = session.run(y_wid_pred_cls, feed_dict=feed_dict_wid)
        cls_pred_col[i:j,:] = session.run(y_col_pred_cls, feed_dict=feed_dict_col)
        cls_pred_ang[i:j] = session.run(y_ang_pred_cls, feed_dict=feed_dict_ang)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_len_true = y_len_int_test
    cls_wid_true = y_wid_int_test
    cls_col_true = y_col_int_test
    cls_ang_true = y_ang_int_test
    #print(cls_ang_true.shape)
    # Create a boolean array whether each image is correctly classified.
    
    correct_len = (cls_len_true == cls_pred_len)
    correct_wid = (cls_wid_true == cls_pred_wid)
    correct_col = (cls_col_true == cls_pred_col)
    correct_ang = (cls_ang_true == cls_pred_ang)
    #print(correct_col.shape)
    #print(correct_ang.shape)
    
    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_len_sum = correct_len.sum()
    correct_wid_sum = correct_wid.sum()
    correct_col_sum = correct_col.sum()
    correct_ang_sum = correct_ang.sum()
    #print(correct_ang_sum)

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc_len = float(correct_len_sum) / num_test
    acc_wid = float(correct_wid_sum) / num_test
    acc_col = float(correct_col_sum) / num_test
    acc_ang = float(correct_ang_sum) / num_test
    
    # Print the accuracy.
    msg = "Accuracy on Test-Set Length: {0:.1%} ({1} / {2}) width: {3:.1%} ({4} / {2}) colour: {5:.1%} ({6} / {2}) angle:{7:.1%} ({8} / {2})"
    print(msg.format(acc_len, correct_len_sum, num_test,acc_wid,correct_wid_sum,acc_col,correct_col_sum,acc_ang,correct_ang_sum))


print_test_accuracy()


optimize(num_iterations=5000)


print_test_accuracy()


#saver = tf.train.Saver()
#save_path = saver.save(session, "/Users/momo/Desktop/assignment2/model1.ckpt",global_step=5000)
#print("Model saved in path: %s" % save_path)


def print_confusion_matrix(x_test,y_test,y_test_cls,y_pred_cls,feed_y,num_classes):
    feed_dict = {x: x_test,
                 feed_y : y_test_cls}
    cls_true = y_test
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict)
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    return(cm)


cm_len = print_confusion_matrix(x_test,y_len_int_test,y_len_int_test,y_len_pred_cls,y_len_true,2)
cm_wid = print_confusion_matrix(x_test,y_wid_int_test,y_wid_int_test,y_wid_pred_cls,y_wid_true,2)
cm_col = print_confusion_matrix(x_test,y_col_int_test,y_col_int_test,y_col_pred_cls,y_col_true,2)
cm_ang = print_confusion_matrix(x_test,y_ang_int_test,y_ang_test,y_ang_pred_cls,y_ang_true,12)



def find_performance_matrix(cm, num_classes):
    performance_matrix = [[0, 0, 0]] * num_classes
    for class_num in range(num_classes):
        performance_matrix[class_num][0] = find_precision(cm, class_num,num_classes)
    for class_num in range(num_classes):
        performance_matrix[class_num][1] = find_recall_rate(cm, class_num,num_classes)
    for class_num in range(num_classes):
        performance_matrix[class_num][2] = find_f_score(performance_matrix[class_num])

    #find mean precision, mean recall rate and mean f score
    for metric in range(3):
        total = 0
        for i in range(num_classes):
            total += performance_matrix[i][metric]

        performance_matrix[num_classes - 1][metric] = total / num_classes

    round_off_performance_matrix(performance_matrix)

    return performance_matrix

def round_off_performance_matrix(performance_matrix):
    for i in range(len(performance_matrix)):
        for j in range(len(performance_matrix[i])):
            performance_matrix[i][j] = round(performance_matrix[i][j], 2)


def find_precision(cm, class_num,num_classes):
    total_samples_classfied_as_class_num = 0
    for i in range(num_classes):
        total_samples_classfied_as_class_num += cm[i][class_num]
    precision_rate = (cm[class_num][class_num] / total_samples_classfied_as_class_num) * 100

    return precision_rate

def find_recall_rate(cm, class_num,num_classes):
    total_samples_in_class = 0
    for j in range(num_classes):
        total_samples_in_class += cm[class_num][j]
    recall_rate = (cm[class_num][class_num] / total_samples_in_class) * 100

    return recall_rate

def find_f_score(array):
    precision = array[0]
    recall = array[1]
    f_score = (precision * recall) / ((precision + recall) / 2)

    return f_score


pm_len = find_performance_matrix(cm_len,2)
print(pm_len)
f_score_len = pm_len[-1][-1]
print(f_score_len)


pm_wid = find_performance_matrix(cm_wid,2)
print(pm_wid)
f_score_wid = pm_wid[-1][-1]
print(f_score_wid)


pm_col = find_performance_matrix(cm_col,2)
print(pm_col)
f_score_col = pm_col[-1][-1]
print(f_score_col)


pm_ang = find_performance_matrix(cm_ang,2)
print(pm_ang)
f_score_ang = pm_ang[-1][-1]
print(f_score_ang)


session.close()
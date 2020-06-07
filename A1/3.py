import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from mnist import MNIST
data = MNIST(data_dir="data/MNIST/")

img_size_flat = data.img_size_flat
img_shape = data.img_shape
num_classes = data.num_classes

x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_cls = tf.placeholder(tf.int64, [None])

def new_FC_layer(x,units,activation,use_bias,name,reuse):
        num_ip = int(x.shape[1])
        num_op = units
        
        #kernel initializer
        #if kernel_initializer==xavier_initializer:
        #    initializer = tf.contrib.layers.xavier_initializer
        #    weight = tf.Variable(initializer([num_ip, num_op]),name='Weight')
        #else:
        
        if name:
            if reuse == True:
                with tf.variable_scope(name,reuse = tf.AUTO_REUSE):
                    weight = tf.get_variable("Weight",[num_ip, num_op],initializer=tf.zeros_initializer)
                    bias = tf.get_variable("Bias",[num_op],initializer=tf.zeros_initializer)
            else:
                with tf.variable_scope(name):
                    weight = tf.get_variable("Weight",[num_ip, num_op],initializer=tf.zeros_initializer)
                    bias = tf.get_variable("Bias",[num_op],initializer=tf.zeros_initializer)
        else:
            weight = tf.get_variable("Weight",[num_ip, num_op],initializer=tf.zeros_initializer)
            bias = tf.get_variable("Bias",[num_op],initializer=tf.zeros_initializer)
        
        if use_bias==False:
            layer = tf.matmul(x,weight)
        else:
            layer = tf.add(tf.matmul(x,weight),bias)
        if activation:
            layer = activation(layer)
        return layer

#h_layer1 = new_FC_layer(x,500,activation = tf.nn.sigmoid,use_bias=True,name='h_layer1',reuse=False)
#h_layer2= new_FC_layer(h_layer1,512,activation = tf.nn.relu,use_bias=True,name ='h_layer2',reuse=False)
#h_layer3= new_FC_layer(h_layer2,512,activation = tf.nn.relu,use_bias=True,name ='h_layer2',reuse=True)
logits = new_FC_layer(x,num_classes,activation = None,use_bias=True,name='op_layer',reuse=True)

#tf.global_variables()

y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, axis=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                           labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session= tf.Session()
session.run(tf.global_variables_initializer())
batch_size = 100
feed_dict_test = {x: data.x_test,
                  y_true: data.y_test,
                  y_true_cls: data.y_test_cls}

def optimize(num_iterations):
    loss_list=[]
    acc_list=[]
    for i in range(num_iterations):
        x_batch, y_true_batch, _ = data.random_batch(batch_size=batch_size)
        
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)
        if i % 100 == 0:
            loss=session.run(cost, feed_dict=feed_dict_train)
            acc = session.run(accuracy, feed_dict=feed_dict_test)
            loss_list.append(loss)
            acc_list.append(acc)
            plt.plot(loss_list)
            plt.savefig("loss_list.png")
            plt.clf()
            plt.plot(acc_list)
            plt.savefig("acc_list.png")
            plt.clf()

def print_accuracy():
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print("Accuracy on test-set: {0:.1%}".format(acc))

def print_confusion_matrix():
    cls_true = data.y_test_cls
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
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
    # plt.show()
    plt.savefig("confusion_matrix.png")
    plt.clf()
    return(cm)

def find_performance_matrix(cm, num_classes):
    performance_matrix = [[0, 0, 0]] * num_classes
    for class_num in range(num_classes):
        performance_matrix[class_num][0] = find_precision(cm, class_num)
    for class_num in range(num_classes):
        performance_matrix[class_num][1] = find_recall_rate(cm, class_num)
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


def find_precision(cm, class_num):
    total_samples_classfied_as_class_num = 0
    for i in range(num_classes):
        total_samples_classfied_as_class_num += cm[i][class_num]
    precision_rate = (cm[class_num][class_num] / total_samples_classfied_as_class_num) * 100

    return precision_rate

def find_recall_rate(cm, class_num):
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
# saver = tf.train.Saver()
# save_path = saver.save(session, "/Users/momo/Desktop/Assignment3/saved_model/model1.ckpt",global_step=1000)
# print("Model saved in path: %s" % save_path)

optimize(num_iterations=1000)
print_accuracy()
cm = print_confusion_matrix()
print(num_classes)
pm = find_performance_matrix(cm, num_classes)
print(pm)
f_score = pm[-1][-1]
print(f_score)
session.close()
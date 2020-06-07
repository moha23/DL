from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import pathlib
import os
import functools
import random

def load_image(path):
  image = tf.read_file(path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [28, 28])
  image /= 255.0
  
  return image

def label_from_image_path(path):
  filename = pathlib.Path(path).stem
  attr = map(lambda x: int(x) + 1, filename.split("_")[:4])

  return functools.reduce(lambda x, y: x*y, attr, 1) - 1

data_root = pathlib.Path('./line-dataset/openCV/')
all_image_paths = list(map(str, filter(lambda x: not x.stem.startswith("."), data_root.glob('*/*.jpg'))))
random.shuffle(all_image_paths)

all_image_labels = list(map(label_from_image_path, all_image_paths))

ds_size = len(all_image_paths)
train_size = int(0.7 * ds_size)
test_size = ds_size - train_size

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = features
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  batch_norm1 = tf.layers.batch_normalization(inputs=conv1, training=is_training)
  pool1 = tf.layers.max_pooling2d(inputs=batch_norm1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  batch_norm2 = tf.layers.batch_normalization(inputs=conv2, training=is_training)
  pool2 = tf.layers.max_pooling2d(inputs=batch_norm2, pool_size=[2, 2], strides=2)

  # # Convolutional Layer #3 and Pooling Layer #3
  conv3 = tf.layers.conv2d(
       inputs=pool2,
       filters=64,
       kernel_size=[3, 3],
       padding="same",
       activation=tf.nn.relu)

  batch_norm3 = tf.layers.batch_normalization(inputs=conv3, training=is_training)

  # Dense Layer
  pool3_flat = tf.reshape(batch_norm3, [-1, 7 * 7 * 64])
  dense1 = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
  #dropout1 = tf.layers.dropout(
  #    inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dense1, units=96)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
      optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=0.1)

      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
  }

  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def training_input_fn():
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_image)
    
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    
    return tf.data.Dataset.zip((image_ds, label_ds)).take(train_size).repeat().batch(100)

def test_input_fn():
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_image)
    
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    
    return tf.data.Dataset.zip((image_ds, label_ds)).skip(train_size).batch(100)

# Create the Estimator
classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./b1")

# Set up logging for predictions
tensors_to_log = {} #{"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

tf.logging.set_verbosity(tf.logging.INFO)

classifier.train(input_fn=training_input_fn, steps=10000)

eval_results = classifier.evaluate(input_fn=test_input_fn)
print(eval_results)

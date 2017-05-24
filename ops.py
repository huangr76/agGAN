from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave
import os

def conv2d(input_map, num_output_channels, size_kernel=5, stride=2, name='conv2d'):
    with tf.variable_scope(name):
        #print(input_map)
        stddev = np.sqrt(2.0 / (np.sqrt(input_map.get_shape()[-1].value * num_output_channels) * size_kernel ** 2))
        kernel = tf.get_variable(
            name='w',
            shape=[size_kernel, size_kernel, input_map.get_shape()[-1], num_output_channels],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=stddev)
        )
        biases = tf.get_variable(
            name='b',
            shape=[num_output_channels],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0)
        )
        conv = tf.nn.conv2d(input_map, kernel, strides=[1, stride, stride, 1], padding='SAME')
        return tf.nn.bias_add(conv, biases)

def fc(input_vector, num_output_length, name='fc'):
    with tf.variable_scope(name):
        #print(input_vector)
        stddev = np.sqrt(1.0 / (np.sqrt(input_vector.get_shape()[-1].value * num_output_length)))
        w = tf.get_variable(
            name='w',
            shape=[input_vector.get_shape()[1], num_output_length],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            name='b',
            shape=[num_output_length],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input_vector, w) + b

def deconv2d(input_map, output_shape, size_kernel=5, stride=2, stddev=0.02, name='deconv2d'):
    with tf.variable_scope(name):
        #print(input_map)
        stddev = np.sqrt(1.0 / (np.sqrt(input_map.get_shape()[-1].value * output_shape[-1]) * size_kernel ** 2))
        # filter : [height, width, output_channels, in_channels]
        kernel = tf.get_variable(
            name='w',
            shape=[size_kernel, size_kernel, output_shape[-1], input_map.get_shape()[-1]],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        biases = tf.get_variable(
            name='b',
            shape=[output_shape[-1]],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0)
        )
        deconv = tf.nn.conv2d_transpose(input_map, kernel, strides=[1, stride, stride, 1], output_shape=output_shape)
        return tf.nn.bias_add(deconv, biases)

def lrelu(logits, leak=0.2):
    return tf.maximum(logits, leak*logits)

def concat_label(x, labels_one_hot, duplicate=1):
    x_shape = x.get_shape().as_list()
    labels_one_hot = tf.tile(labels_one_hot, [1, duplicate])
    labels_one_hot_shape = labels_one_hot.get_shape().as_list()
    if len(x_shape) == 2:
        return tf.concat([x, labels_one_hot], axis=1)
    elif len(x_shape) == 4:
        labels_one_hot = tf.reshape(labels_one_hot, [x_shape[0], 1, 1, labels_one_hot_shape[-1]])
        return tf.concat([x, labels_one_hot*tf.ones([x_shape[0], x_shape[1], x_shape[2], labels_one_hot_shape[-1]])])

def get_dataset(img_dir, list_file):
    """return a list that each row contain a image_path, id and age label"""
    dataset = []
    with open(list_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            split = line.split(' ')
            img_name = split[0]
            id = split[1]
            age = int(split[2])
            if 16 <= age <= 20:
                age = 0
            elif 21 <= age <= 30:
                age = 1
            elif 31 <= age <= 40:
                age = 2
            elif 41 <= age <= 50:
                age = 3
            elif 51 <= age <= 60:
                age = 4
            elif 61 <= age <= 70:
                age = 5
            else:
                age = 6
            img_path = os.path.join(img_dir, img_name)
            dataset.append(img_path + ' ' + id + ' ' + str(age))
        #print(dataset)
    return dataset

def transform(image):
    #[0, 1]->[-1, 1]
    with tf.name_scope("preprocess"):
        return image * 2 - 1

        
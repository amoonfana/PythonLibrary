# -*- coding: utf-8 -*-
import Reader as rd
import tensorflow as tf

def readData(address):
    data = rd.readBinFile(address)
    data = data.reshape(data.shape[0], data.shape[1]*data.shape[2]) / 255
    
    return data

def readLabel(address):
    label = rd.onehotEncoder(rd.readBinFile(address).astype(int), 10)
    
    return label

train_data = readData('MNIST_data/train-images.idx3-ubyte')
train_label = readLabel('MNIST_data/train-labels.idx1-ubyte')
test_data = readData('MNIST_data/t10k-images.idx3-ubyte')
test_label = readLabel('MNIST_data/t10k-labels.idx1-ubyte')

#Build model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
#y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Train model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(400):
        sess.run(train_step, feed_dict={x: train_data, y_: train_label})

        if i % 10 == 0:
            print(sess.run(accuracy, feed_dict={x: test_data, y_: test_label}))
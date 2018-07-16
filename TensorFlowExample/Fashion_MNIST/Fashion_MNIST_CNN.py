# -*- coding: utf-8 -*-
import Reader as rd
import matplotlib.pyplot as pyplot
import tensorflow as tf

def readData(address):
    data = rd.readBinFile(address)
    data = data.reshape(data.shape[0], data.shape[1]*data.shape[2]) / 255
    
    return data

def readLabel(address):
    label = rd.onehotEncoder(rd.readBinFile(address).astype(int), 10)
    
    return label

#Weight Initialization
def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

#Convolution and Pooling
def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def build_network():
      x = tf.placeholder(tf.float32, [None, 784])
      y_ = tf.placeholder(tf.float32, [None, 10])
      
      #First Convolutional Layer
      W_conv1 = weight_variable([5, 5, 1, 32])
      b_conv1 = bias_variable([32])
      
      x_image = tf.reshape(x, [-1, 28, 28, 1])
      h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
      h_pool1 = max_pool_2x2(h_conv1)
      
      #Second Convolutional Layer
      W_conv2 = weight_variable([5, 5, 32, 64])
      b_conv2 = bias_variable([64])
      
      h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
      h_pool2 = max_pool_2x2(h_conv2)
      
      #Densely Connected Layer
      W_fc1 = weight_variable([7*7*64, 1024])
      b_fc1 = bias_variable([1024])
      
      h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

      #Dropout
      keep_prob = tf.placeholder(tf.float32)
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

      #Readout Layer
      W_fc2 = weight_variable([1024, 10])
      b_fc2 = bias_variable([10])
      
      y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
      
      return (x, y_, y_conv, keep_prob)

def evaluate():
      train_data = readData('Fashion_MNIST_data/train-images-idx3-ubyte')
      train_label = readLabel('Fashion_MNIST_data/train-labels-idx1-ubyte')
      test_data = readData('Fashion_MNIST_data/t10k-images-idx3-ubyte')
      test_label = readLabel('Fashion_MNIST_data/t10k-labels-idx1-ubyte')

      #Train and Evaluate the Model
      x, y_, y_conv, keep_prob = build_network()
      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
      train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
      correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      
      res=[]
      i_s = 0
      i_e = 99
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(10000):
          batch_data = train_data[i_s:i_e]
          batch_label = train_label[i_s:i_e]
          i_s = (i_s+100) % 60000
          i_e = (i_e+100) % 60000
          sess.run(train_step, feed_dict={x: batch_data, y_: batch_label, keep_prob: 0.5})
          
          if i % 100 == 0:
            test_accuracy = accuracy.eval(feed_dict={x: test_data, y_: test_label, keep_prob: 1.0})
            print('step %d, test accuracy %g' % (i, test_accuracy))
            res.append(test_accuracy)
        
        #Draw test accuracy
        pyplot.plot(res)
        pyplot.ylim(0.75, 1)
        pyplot.xlim(0,120)
        pyplot.xlabel('batch (x 100)')
        pyplot.ylabel('test accuracy')

evaluate()
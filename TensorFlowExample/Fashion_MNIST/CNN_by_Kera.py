# -*- coding: utf-8 -*-
import Data as rd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def readData(address):
    data = rd.readBinFile(address) / 255
#    data = data.reshape(data.shape[0], data.shape[1]*data.shape[2]) / 255
    
    return data

def readLabel(address):
    label = rd.readBinFile(address)
    
    return label

def draw(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
#    plt.gca().grid(False)
    
def drawSample(images):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
#        plt.grid('off')
        plt.imshow(train_data[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_label[i]])
    
def drawPredictionSample(test_data, test_label):
    predictions = model.predict(test_data)
    
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid('off')
        plt.imshow(test_data[i], cmap=plt.cm.binary)
        predicted_label = np.argmax(predictions[i])
        true_label = test_label[i]
        if predicted_label == true_label:
          color = 'green'
        else:
          color = 'red'
        plt.xlabel("{} ({})".format(class_names[predicted_label],
                                      class_names[true_label]),
                                      color=color)

if __name__ == '__main__':
    train_data = readData('Fashion_MNIST_data/train-images-idx3-ubyte')
    train_label = readLabel('Fashion_MNIST_data/train-labels-idx1-ubyte')
    test_data = readData('Fashion_MNIST_data/t10k-images-idx3-ubyte')
    test_label = readLabel('Fashion_MNIST_data/t10k-labels-idx1-ubyte')
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    draw(train_data[0])
    drawSample(train_data)

    model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(10, activation=tf.nn.softmax)
            ])
    
    model.compile(
            optimizer=tf.train.AdamOptimizer(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
            )
    
    model.fit(train_data, train_label, epochs=5)
    
    test_loss, test_acc = model.evaluate(test_data, test_label)

    print('Test accuracy:', test_acc)
    
    drawPredictionSample(test_data, test_label)
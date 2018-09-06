# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(path='Data/boston_housing.npz', test_split=0.2, seed=113):
  f = np.load(path)
  x = f['x']
  y = f['y']
  f.close()

  np.random.seed(seed)
  indices = np.arange(len(x))
  np.random.shuffle(indices)
  x = x[indices]
  y = y[indices]

  x_train = np.array(x[:int(len(x) * (1 - test_split))])
  y_train = np.array(y[:int(len(x) * (1 - test_split))])
  x_test = np.array(x[int(len(x) * (1 - test_split)):])
  y_test = np.array(y[int(len(x) * (1 - test_split)):])
  return (x_train, y_train), (x_test, y_test)

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, 
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  model.compile(loss='mse',
                optimizer = tf.train.RMSPropOptimizer(0.001),
                metrics=['mae'])
  return model

# Display training progress by printing a single dot for each completed epoch.
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label = 'Val loss')
    plt.legend()
    plt.ylim([0,5])

if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = load_data()
    
    # Shuffle the training set
    order = np.argsort(np.random.random(train_labels.shape))
    train_data = train_data[order]
    train_labels = train_labels[order]
    
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                    'TAX', 'PTRATIO', 'B', 'LSTAT']
    
    df = pd.DataFrame(train_data, columns=column_names)
    
    # Test data is *not* used when calculating the mean and std.
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    
    model = build_model()
    model.summary()

    # The patience parameter is the amount of epochs to check for improvement.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    
    # Store training stats
    history = model.fit(train_data, train_labels, epochs=500,
                        validation_split=0.2, verbose=0,
                        callbacks=[early_stop, PrintDot()])
    
    plot_history(history)
    
    [loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

    print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))
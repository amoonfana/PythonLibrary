# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 14:48:27 2018

@author: Vera
"""

import numpy as np
import matplotlib.pyplot as plt

def computeCost(X, Y, W):
      m = np.size(Y);
      Z = np.dot(X,W) - Y;
      j = np.dot(Z.T,Z) / (2*m);
      
      return j;

def gradientDescent(X, Y, W, lr, iterations):
      m = np.size(Y);
      J = np.zeros([iterations, 1]);
      
      for i in range(iterations):
            z = np.dot(X.T, np.dot(X,W) - Y);
            W = W - lr/m*z;
            J[i] = computeCost(X, Y, W);

      return (W,J)

data = np.loadtxt('data.txt', delimiter=',');

m = np.size(data[:, 1]);
X = np.c_[np.ones([m, 1]), data[:,0]];
W = np.zeros([2, 1]);
Y = data[:, 1].reshape([m, 1]);

iterations = 1500;
lr = 0.01;

(W,J) = gradientDescent(X, Y, W, lr, iterations);

plt.figure()
plt.plot(data[:,0], data[:,1],'rx',markersize=8);
plt.plot(data[:,0], np.dot(X,W),'-');

plt.figure()
plt.plot(range(1500), J,'-');
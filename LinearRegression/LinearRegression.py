# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def cost_gradient(W, X, Y, n):
      Z = np.dot(X,W) - Y;
      G = np.dot(X.T, Z) / n;
      j = np.dot(Z.T,Z) / (2*n);
      
      return (j, G);

def gradientDescent(W, X, Y, lr, iterations):
      n = np.size(Y);
      J = np.zeros([iterations, 1]);
      
      (J[0], G) = cost_gradient(W, X, Y, n);
      for i in range(iterations):
            W = W - lr*G;
            (J[i], G) = cost_gradient(W, X, Y, n);

      return (W,J);

data = np.loadtxt('data.txt', delimiter=',');

n = np.size(data[:, 1]);
W = np.zeros([2, 1]);
X = np.c_[np.ones([n, 1]), data[:,0]];
Y = data[:, 1].reshape([n, 1]);

iterations = 1500;
lr = 0.01;

(W,J) = gradientDescent(W, X, Y, lr, iterations);

#Draw figure
plt.figure();
plt.plot(data[:,0], data[:,1],'rx');
plt.plot(data[:,0], np.dot(X,W));

plt.figure();
plt.plot(range(iterations), J);
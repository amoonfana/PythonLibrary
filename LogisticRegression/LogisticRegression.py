# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(Z):
      return 1.0/(1.0+np.exp(-Z));

def regularized_cost_gradient(W, X, Y, pnt, n):
      W1 = W.copy();
      W1[0] = 0;
      
      S = sigmoid(np.dot(X, W));
      j = -(np.dot(Y.T, np.log(S)) + np.dot((1-Y).T, np.log(1-S)) + np.dot(W1.T, W)*pnt/2) / n;
      G = (np.dot(X.T, S - Y.reshape(n,1)) + pnt*W1) / n;
      
      return (j, G);

def gradientDescent(W, X, Y, pnt, lr, iterations):
      n = np.size(Y);
      J = np.zeros([iterations, 1]);
      
      (J[0], G) = regularized_cost_gradient(W, X, Y, pnt, n);
      for i in range(iterations):
            W = W - lr*G;
            (J[i], G) = regularized_cost_gradient(W, X, Y, pnt, n);

      return (W,J);

data = np.loadtxt('data1.txt', delimiter=',');

W = np.zeros([data.shape[1], 1]);
X = np.c_[np.ones([data.shape[0], 1]), data[:,0:-1]];
Y = data[:,-1];

iterations = 200000;
lr = 0.001;
pnt = 0.1;

(W,J) = gradientDescent(W, X, Y, pnt, lr, iterations);

#Draw figure
plt.figure();
pi=np.where(Y==1);
ni=np.where(Y==0);
plt.plot(data[pi,0], data[pi,1], 'rx');
plt.plot(data[ni,0], data[ni,1], 'bo');

plot_x = [min(X[:,1]),  max(X[:,1])];
plot_y = (-1/W[2])*(np.multiply(W[1],plot_x) + W[0]);

plt.plot(plot_x, plot_y);

plt.figure();
plt.plot(range(iterations), J);
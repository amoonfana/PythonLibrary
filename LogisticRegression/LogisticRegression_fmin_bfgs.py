# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def sigmoid(Z):
      return 1.0/(1.0+np.exp(-Z));

#For bfgs optimization
def cost(W, X, Y, pnt, n):
      W1 = W.copy();
      W1[0] = 0;
      
      S = sigmoid(np.dot(X, W));
      j = -(np.dot(Y.T, np.log(S)) + np.dot((1-Y).T, np.log(1-S)) + np.dot(W1.T, W)*pnt/2) / n;
           
      return j;

#For bfgs optimization
def gradient(W, X, Y, pnt, n):
      W1 = W.copy();
      W1[0] = 0;
      
      S = sigmoid(np.dot(X, W));
      G = (np.dot(X.T, S - Y) + pnt*W1) / n;
      
      return G;

data = np.loadtxt('data1.txt', delimiter=',');

W = np.zeros([data.shape[1], 1]);
X = np.c_[np.ones([data.shape[0], 1]), data[:,0:-1]];
Y = data[:,-1];

iterations = 200000;
pnt = 0.1;

W = opt.fmin_bfgs(cost, W, fprime=gradient, args=(X,Y,pnt,np.size(Y)));

#Draw figure
plt.figure();
pi=np.where(Y==1);
ni=np.where(Y==0);
plt.plot(data[pi,0], data[pi,1], 'rx');
plt.plot(data[ni,0], data[ni,1], 'bo');

plot_x = [min(X[:,1]),  max(X[:,1])];
plot_y = (-1/W[2])*(np.multiply(W[1],plot_x) + W[0]);

plt.plot(plot_x, plot_y);
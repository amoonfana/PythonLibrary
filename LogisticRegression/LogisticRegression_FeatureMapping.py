# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def sigmoid(Z):
      return 1.0/(1.0+np.exp(-Z));

##For bfgs optimization
#def cost(W, X, Y, pnt, n):
#      W1 = W.copy();
#      W1[0] = 0;
#      
#      S = sigmoid(np.dot(X, W));
#      j = -(np.dot(Y.T, np.log(S)) + np.dot((1-Y).T, np.log(1-S)) + np.dot(W1.T, W)*pnt/2) / n;
#           
#      return j;
#
##For bfgs optimization
#def gradient(W, X, Y, pnt, n):
#      W1 = W.copy();
#      W1[0] = 0;
#      
#      S = sigmoid(np.dot(X, W));
#      G = (np.dot(X.T, S - Y) + pnt*W1) / n;
#      
#      return G;

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

def mapFeature(X1,X2):
    degree = 2;
    out = np.ones((X1.shape[0],1));
    
    for i in range(1,degree+1): 
        for j in range(i+1):
            temp = np.power(X1,i-j)*np.power(X2,j);
            out = np.c_[out, temp.reshape(-1,1)];
    return out;

def plot(W, X, Y):
    pi = np.where(Y==1);
    ni = np.where(Y==0);
    
    plt.figure();
    plt.plot(X[pi,0],X[pi,1],'rx');
    plt.plot(X[ni,0],X[ni,1],'bo');
    
    u = np.linspace(-1,1,50);
    v = np.linspace(-1,1,50);
    
    z = np.zeros((len(u),len(v)));
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.dot(mapFeature(u[i].reshape(1,-1),v[j].reshape(1,-1)), W);
    
    plt.contour(u,v,z.T,[0,0.01],linewidth=2.0);
    plt.show();

data = np.loadtxt('data2.txt', delimiter=',');

X = data[:,0:-1];
X = mapFeature(X[:,0], X[:,1]);
Y = data[:,-1];
W = np.zeros([X.shape[1], 1]);

iterations = 600;
lr = 1;
pnt = 0.1;

(W,J) = gradientDescent(W, X, Y, pnt, lr, iterations);
#W = opt.fmin_bfgs(cost, W, fprime=gradient, args=(X,Y,pnt,np.size(Y)));

plot(W, data[:,0:-1], Y);

plt.figure();
plt.plot(range(iterations), J);
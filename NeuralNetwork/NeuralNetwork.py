# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio
import NN_Utils as nnu

#Xavier Initialization
def initWeights(M):
      l = len(M)
      W = []
      B = []
      
      for i in range(1, l):
            W.append(np.random.randn(M[i-1], M[i]) / np.sqrt(M[i-1]))
            B.append(np.zeros([1, M[i]]))
            
      return W, B

#Forward propagation
def sigmoidForward(Z):
      A = 1/(1+np.exp(-Z))
      return A

def linearForward(A_in, W_in, B_in):
      Z = np.dot(A_in, W_in) + B_in
      return Z

def layerForward(A_in, W_in, B_in):
      Z = linearForward(A_in, W_in, B_in)
      A = sigmoidForward(Z)

      return A

def networkForward(X, W, B):
      l = len(W)
      A = [None for i in range(l+1)]
      A[0] = X

      for i in range(0, l):
            A[i+1] = layerForward(A[i], W[i], B[i])

      return A
#--------------------------

#Backward propagation
def sigmoidBackward(dA, A):
      dZ = dA * A * (1-A)

      return dZ

def linearBackward(dZ, A_in, W_in):
      n = A_in.shape[0]
      
      dW_in = (np.dot(A_in.T, dZ)) / n
      dB_in = np.sum(dZ, axis=0, keepdims=True) / n
      dA_in = np.dot(dZ, W_in.T)
      
      return dA_in, dW_in, dB_in

def layerBackward(dA, A, A_in, W_in):
      dZ = sigmoidBackward(dA, A)
      dA_in, dW_in, dB_in = linearBackward(dZ, A_in, W_in)

      return dA_in, dW_in, dB_in

def networkBackward(Y, A, W):
      l = len(W)
      dW = [None for i in range(l)]
      dB = [None for i in range(l)]
      
      dA = -Y/A[l] + (1-Y)/(1-A[l])
      for i in reversed(range(l)):
            dA, dW[i], dB[i] = layerBackward(dA, A[i+1], A[i], W[i])

      return dW, dB
#--------------------------

#Update weights by gradient descent
def updateWeights(W, B, dW, dB, lr):
      l = len(W)

      for i in range(l):
            W[i] = W[i] - lr*dW[i]
            B[i] = B[i] - lr*dB[i]

      return W, B

#Compute regularized cost function
def cost(A_l, Y, W):
      n = Y.shape[0]
      
      logprobs = np.log(A_l)*Y + np.log(1-A_l)*(1-Y)
      c = -1./n * np.nansum(logprobs)

      return c

def train(X, Y, M, lr = 0.0075, iterations = 3000):
      costs = []
      W, B = initWeights(M)

      for i in range(iterations):
            A = networkForward(X, W, B)
            c = cost(A[-1], Y, W)
            dW, dB = networkBackward(Y, A, W)
            W, B = updateWeights(W, B, dW, dB, lr)

            if i % 100 == 0:
                  print("Cost after iteration %i: %f" %(i, c))
                  costs.append(c)

      return W, B, costs

def predict(X, W, B, ny):
      Y_out = np.zeros([X.shape[0], ny])
      
      A = networkForward(X, W, B)
      idx = np.argmax(A[-1], axis=1)
      Y_out[range(Y.shape[0]),idx] = 1
      
      return Y_out

def test(Y, X, W, B):
      Y_out = predict(X, W, B, Y.shape[1])
      acc = np.sum(Y_out*Y) / Y.shape[0]
      print("Test accuracy is: %f" %(acc))
      
      return acc

data = spio.loadmat("data.mat")
(n, m) = data['X'].shape

idx = np.random.randint(0, n, 100)
nnu.drawData(data['X'][idx,:])

X = data['X']
Y = nnu.onehotEncoder(data['y'], 10)
M = [400, 25, 10]

W, B, costs = train(X, Y, M, lr = 1)
test(Y, X, W, B)
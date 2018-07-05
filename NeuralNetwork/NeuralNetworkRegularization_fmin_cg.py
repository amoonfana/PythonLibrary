# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio
import scipy.optimize as opt
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

def linearBackwardRegularized(dZ, A_in, W_in, pnt):
      n = A_in.shape[0]
      
      dW_in = (np.dot(A_in.T, dZ) +  pnt*W_in) / n
      dB_in = np.sum(dZ, axis=0, keepdims=True) / n
      dA_in = np.dot(dZ, W_in.T)
      
      return dA_in, dW_in, dB_in

def layerBackwardRegularized(dA, A, A_in, W_in, pnt):
      dZ = sigmoidBackward(dA, A)
      dA_in, dW_in, dB_in = linearBackwardRegularized(dZ, A_in, W_in, pnt)

      return dA_in, dW_in, dB_in

def networkBackwardRegularized(Y, A, W, pnt):
      l = len(W)
      dW = [None for i in range(l)]
      dB = [None for i in range(l)]
      
      dA = -Y/A[l] + (1-Y)/(1-A[l])
      for i in reversed(range(l)):
            dA, dW[i], dB[i] = layerBackwardRegularized(dA, A[i+1], A[i], W[i], pnt)

      return dW, dB
#--------------------------

#Compute regularized cost function
def costRegularized(A_l, Y, W, pnt):
      n = Y.shape[0]
      l = len(W)
      
      c_regularization = 0
      for i in range(l):
            c_regularization = c_regularization + pnt/(2*n) * np.sum(np.square(W[i]))
      
      logprobs = np.log(A_l)*Y + np.log(1-A_l)*(1-Y)
      c_crossEntropy = -1./n * np.nansum(logprobs)
      
      c =  c_crossEntropy + c_regularization

      return c

def predict(X, W, B, ny):
      Y_out = np.zeros([X.shape[0], ny])
      
      A = networkForward(X, W, B)
      idx = np.argmax(A[-1], axis=1)
      Y_out[range(Y.shape[0]),idx] = 1
      
      return Y_out

def testRegularized(Y, X, W, B):
      Y_out = predict(X, W, B, Y.shape[1])
      acc = np.sum(Y_out*Y) / Y.shape[0]
      print("Test accuracy is: %f" %(acc))
      
      return acc

def nnCostFunctionRegularized(WB, X, Y, pnt):
      W = [WB[:10000].reshape(400,25), WB[10000:10250].reshape(25,10)]
      B = [WB[10250:10275].reshape(1,25), WB[10275:].reshape(1,10)]
      
      A = networkForward(X, W, B)
      c = costRegularized(A[-1], Y, W, pnt)
      
      return np.ravel(c)

def nnGradientRegularized(WB, X, Y, pnt):
      W = [WB[:10000].reshape(400,25), WB[10000:10250].reshape(25,10)]
      B = [WB[10250:10275].reshape(1,25), WB[10275:].reshape(1,10)]
      
      A = networkForward(X, W, B)
      dW, dB = networkBackwardRegularized(Y, A, W, pnt)
      
      dWB = np.vstack((dW[0].reshape(-1,1), dW[1].reshape(-1,1), dB[0].reshape(-1,1), dB[1].reshape(-1,1)))
      
      return np.ravel(dWB)

data = spio.loadmat("data.mat")
(n, m) = data['X'].shape

idx = np.random.randint(0, n, 100)
nnu.drawData(data['X'][idx,:])

X = data['X']
Y = nnu.onehotEncoder(data['y'], 10)
M = [400, 25, 10]

W, B = initWeights(M)
WB = np.vstack((W[0].reshape(-1,1), W[1].reshape(-1,1), B[0].reshape(-1,1), B[1].reshape(-1,1)))
WB = opt.fmin_cg(nnCostFunctionRegularized, WB, fprime = nnGradientRegularized, args=(X, Y, 1), maxiter=100)

W = [WB[:10000].reshape(400,25), WB[10000:10250].reshape(25,10)]
B = [WB[10250:10275].reshape(1,25), WB[10275:].reshape(1,10)]
testRegularized(Y, X, W, B)

## plot the cost
#plt.plot(np.squeeze(costs))
#plt.ylabel('cost')
#plt.xlabel('iterations (per tens)')
#plt.title("Learning rate =" + str(lr))
#plt.show()
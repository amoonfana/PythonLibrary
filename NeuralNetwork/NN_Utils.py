# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

def drawData(data):
      col = 0
      (n,m) = data.shape
      width = np.int32(np.round(np.sqrt(m)))
      height = np.int32(m/width);
      rows = np.int32(np.floor(np.sqrt(n)))
      cols = np.int32(np.ceil(n/rows))
      pad = 1
      matrix = -np.ones((pad + rows*(height+pad), pad + cols*(width+pad)))
      
      for i in range(rows):
            for j in range(cols):
                  if col >= n:
                        break;
                  matrix[(pad + i*(height+pad)):(pad + i*(height+pad) + height), (pad + j*(width+pad)):(pad + j*(width+pad) + width)] = data[col,:].reshape(height, width, order="F")
                  col += 1
            if col >= n:
                  break;
            
      plt.imshow(matrix, cmap='gray')
      plt.axis('off')
      plt.show()

def onehotEncoder(Y, ny):
      return np.eye(ny)[Y.flatten()]

#def gradientCheck(X, Y, W, epsilon = 1e-7):
#      l = len(W)
#      G_approx = [None for i in range(l)]
#      
#      A_l = networkForward(X, W)
#      G = networkBackward(Y, A_l, W)
#      
#      for i in range(l):
#            G_approx[i] = np.zeros_like(W[i])
#            
#            for j in range(W[i].shape[0]):
#                  print([i,j])
#                  for k in range(W[i].shape[1]):
#                        W_plus = W
#                        W_minus = W
#                        W_plus[i][j,k] = W_plus[i][j,k] + epsilon
#                        W_minus[i][j,k] = W_minus[i][j,k] - epsilon
#
#                        A_plus = networkForward(X, W_plus)
#                        A_minus = networkForward(X, W_minus)
#                        c_plus = cost(A_plus[-1], Y)
#                        c_minus = cost(A_minus[-1], Y)
#                        G_approx[i][j,k] = (c_plus - c_minus)/(2*epsilon)
#
#                        numerator = np.abs(G[i][j,k] - G_approx[i][j,k])
#                        denominator = G[i][j,k] + G_approx[i][j,k]
#                        difference = numerator/denominator
#
#                        if difference > 1e-7:
#                              print ("The gradient is wrong on: ", [i,j,k, difference])
#
#      numerator = np.sqrt(np.dot((G - G_approx).T, (G - G_approx)))
#      denominator = np.sqrt(np.dot(G.T, G)) + np.sqrt(np.dot(G_approx.T, G_approx))
#      difference = numerator/denominator
#
#      if difference > 2e-7:
#            print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
#      else:
#            print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
#
#      return difference
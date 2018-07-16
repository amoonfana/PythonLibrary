# -*- coding: utf-8 -*-
import numpy as np

def initWeights(M):
      l = len(M)
      W = []
      b = []
      
      for i in range(0, l):
            #f = M[i][0], n_c_in = M[i][1], n_c = M[i][2]
            W.append(np.random.randn(M[i][0], M[i][0], M[i][1], M[i][2]) / np.sqrt(M[i][0]*M[i][0]*M[i][1]))
            b.append(np.zeros([1, 1, 1, M[i][2]]))
            
      return W, b

def zeroPad(X, pad):
      X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values = (0))

      return X_pad

def conv2dForward(A_in, W, b, hparameters):
    # Retrieve dimensions from A_prev's shape (≈1 line)  
    (m, n_h_in, n_w_in, n_c_in) = A_in.shape
    
    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_c_in, n_c) = W.shape
    
    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    n_h = np.int((n_h_in - f + 2*pad)/stride) + 1
    n_w = np.int((n_w_in - f + 2*pad)/stride) + 1
    
    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros([m, n_h, n_w, n_c])
    
    # Create A_prev_pad by padding A_prev
    A_in_pad = zeroPad(A_in, pad)
    
    for i in range(m):
        a_in_pad = A_in_pad[i]
        
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    # Find the corners of the current "slice" (≈4 lines)
                    hi = h*stride
                    wi = w*stride
                    
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_in_sliced = a_in_pad[hi:(hi+f), wi:(wi+f), :]
                    
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                    Z[i, h, w, c] = np.sum(W[:,:,:,c]*a_in_sliced) + np.asscalar(b[:,:,:,c])
    
    # Save information in "cache" for the backprop
    cache = (A_in, W, b, hparameters)
    
    return Z, cache

def pool2dForward(A_prev, hparameters):    
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve hyperparameters from "hparameters"
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Define the dimensions of the output
    n_H = np.int(1 + (n_H_prev - f) / stride)
    n_W = np.int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    # Initialize output matrix A
    A = np.zeros((m, n_H, n_W, n_C))              
    
    for i in range(m):                         # loop over the training examples
        for h in range(n_H):                     # loop on the vertical axis of the output volume
            for w in range(n_W):                 # loop on the horizontal axis of the output volume
                for c in range (n_C):            # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    hi = h*stride
                    wi = w*stride
                    
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                    a_prev_slice = A_prev[i, hi:(hi+f), wi:(wi+f), c]
                    
                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                    A[i, h, w, c] = np.max(a_prev_slice)
    
    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    
    return A, cache

def fcForward(A_in, W_in, b_in):
      Z = np.dot(A_in, W_in) + b_in
      return Z

def cnn2dForward(X, W, b):
      

def conv2dBackward(dZ, cache):
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache
    
    # Retrieve dimensions from A_prev's shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    # Retrieve dimensions from W's shape
    (f, f, n_C_prev, n_C) = W.shape
    
    # Retrieve information from "hparameters"
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Retrieve dimensions from dZ's shape
    (m, n_H, n_W, n_C) = dZ.shape
    
    # Initialize dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros_like(A_prev)                           
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    # Pad A_prev and dA_prev
    A_prev_pad = zeroPad(A_prev, pad)
    dA_prev_pad = zeroPad(dA_prev, pad)
    
    print(W.shape)
    print(dZ.shape)
    for i in range(m):                       # loop over the training examples
        
        # select ith training example from A_prev_pad and dA_prev_pad
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        
        for h in range(n_H):                   # loop over vertical axis of the output volume
            for w in range(n_W):               # loop over horizontal axis of the output volume
                for c in range(n_C):           # loop over the channels of the output volume
                    
                    # Find the corners of the current "slice"
                    vert_start = h*stride
                    vert_end = h*stride + f
                    horiz_start = w*stride
                    horiz_end = w*stride + f
                    
                    # Use the corners to define the slice from a_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

                    # Update gradients for the window and the filter's parameters using the code formulas given above
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
                    
        # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    
    return dA_prev, dW, db

def pool2dBackward(dA, cache):   
    # Retrieve information from cache (≈1 line)
    (A_prev, hparameters) = cache
    
    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C =  dA.shape
    
    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    
    for i in range(m):                       # loop over the training examples
        
        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i]
        
        for h in range(n_H):                   # loop on the vertical axis
            for w in range(n_W):               # loop on the horizontal axis
                for c in range(n_C):           # loop over the channels (depth)
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h*stride
                    vert_end = h*stride + f
                    horiz_start = w*stride
                    horiz_end = w*stride + f
                    
                    # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                    a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                    # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                    dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += dA[i, h, w, c] * (a_prev_slice==np.max(a_prev_slice))
    
    return dA_prev
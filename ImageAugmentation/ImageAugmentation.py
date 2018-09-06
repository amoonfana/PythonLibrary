# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.preprocessing.image as img
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize, rotate

def imageFlip(image):
    image_flip = image
    
    flip_vertical = np.random.rand()
    if flip_vertical > 0.5:
        image_flip = image_flip[::-1,:]
    
    flip_horizontal = np.random.rand()
    if flip_horizontal > 0.5:
        image_flip = image_flip[:,::-1]
    
    return image_flip

def imageRotate(image):
    angle =np.random.rand()*90
    image_rotate = rotate(image, angle, mode = 'edge')
    
    return image_rotate

def imageCrop(image, crop_rate):
    image_crop = image
    
    idx_up = 0
    crop_up = np.random.rand()
    if crop_up > 0.5:
        idx_up = int(image_crop.shape[0]*crop_rate*np.random.rand())
        image_crop = image_crop[idx_up:]
    
    idx_down = 0
    crop_down = np.random.rand()
    if crop_down > 0.5:
        idx_down = int(image_crop.shape[0]*crop_rate*np.random.rand())
        image_crop = image_crop[0:(image_crop.shape[0]-idx_down)]
    
    idx_left = 0
    crop_left = np.random.rand()
    if crop_left > 0.5:
        idx_left = int(image_crop.shape[1]*crop_rate*np.random.rand())
        image_crop = image_crop[:,idx_left:]
    
    idx_right = 0
    crop_right = np.random.rand()
    if crop_right > 0.5:
        idx_right = int(image_crop.shape[1]*crop_rate*np.random.rand())
        image_crop = image_crop[:,0:(image_crop.shape[1]-idx_right)]
    
    pad_or_resize = np.random.rand()
    if pad_or_resize > 0.2:
        pad_height = [0, 0]
        pad_idx = np.random.randint(2, size=2)
        pad_height[pad_idx[0]] += idx_up
        pad_height[pad_idx[1]] += idx_down
        
        pad_width = [0, 0]
        pad_idx = np.random.randint(2, size=2)
        pad_width[pad_idx[0]] += idx_left
        pad_width[pad_idx[1]] += idx_right
        
        image_crop = np.pad(image_crop, (pad_height, pad_width, (0,0)), mode = 'edge')
    else:
        image_crop = resize(image_crop, (image.shape[0], image.shape[1]))
    
    return image_crop

def imageGenerate(image, size):
    image_gen = resize(image, size)
    image_gen = imageFlip(image_gen)
    image_gen = imageRotate(image_gen)
    image_gen = imageCrop(image_gen, 0.2)
    
    return image_gen

#train_folders = os.listdir('Data/training')
#n_train = len(train_folders)
#
#train_files = [None]*n_train
#for i in range(n_train):
#    train_folders[i] = 'Data/training/' + train_folders[i] + '/'
#    train_files[i] = os.listdir(train_folders[i])
#
#train_images = []
#for i in range(n_train):
#    for j in range(len(train_files[i])):
#        image = imread(train_folders[i] + train_files[i][j])
#        image = resize(image, (64,64))
#        train_images.append(image)

#test_folders = os.listdir('Data/testing')
#n_test = len(test_folders)
#
#test_files = [None]*n_test
#for i in range(n_test):
#    test_folders[i] = 'Data/testing/' + test_folders[i] + '/'
#    test_files[i] = os.listdir(test_folders[i])
#
#test_addr = []
#for i in range(n_test):
#    for j in range(len(test_files[i])):
#        test_addr.append(test_folders[i] + test_files[i][j])
    
image = imread('Data/training/n8/n8070.jpg')
image1 = imageGenerate(image, (64,64))
plt.figure(figsize = (image1.shape[0]*0.1,image1.shape[1]*0.1))
plt.imshow(image1)
plt.show()


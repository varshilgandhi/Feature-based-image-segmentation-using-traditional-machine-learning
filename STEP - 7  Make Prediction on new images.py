# -*- coding: utf-8 -*-
"""
Created on Sat May 15 05:24:38 2021

@author: abc
"""
#STEP : 7 MAKE PREDICTION ON NEW IMAGES

import numpy as np
import cv2
import pandas as pd

def feature_extraction(img):
    df = pd.DataFrame()
    
#ALL FEATURES GENERATED MUST MATCH THE WAY FEATURES ARE GENERATED FOR TRAINING
#FEATURE 1  is our original image pixes

    img2 = img.reshape(-1)
    df['Original Image'] = img2
    
    #GENERATE GABOR FEATURES
    num = 1
    kernels = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1,3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
                    #print(theta, sigma, lamda, frequency)
                    
                    gabor_label = 'Gabor' + str(num)
                    #print(gabor_label)
                    ksize = 9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
                    #Now filter image and add values to new column
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img #Modify this to add column for each gabor
                    num += 1
                
                
###########################################################################################


#Generate Other Features and add them to the data frame
#Feature 3 is canny edge

    edges = cv2.Canny(img, 100, 200)  #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe
    
    
    from skimage.filters import roberts, sobel, scharr, prewitt
    
#Feature 4 is Roberts edge
    
    #ROBERTS EDGE
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1
    
    #SOBEL
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1
    
    #SCHARR
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1
    
    #PREWITT
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1
    
    #GAUSSIAN with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1
    
    #GAUSSIAN with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3
    
    #MEDIAN with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1
    
    
    #VARIANCE with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img1  #Add column to original dataframe
    
    
    return df

###############################################################################


#Applying trained model to segment multiple files

import pickle
from matplotlib import pyplot as plt

filename = "sandastone_model_new"
loaded_model = pickle.load(open(filename, 'rb'))

path = "C:\\Users\\abc\\Desktop\\Digital Sreeni\\Image Segmentation using Traditional machine learning"

import os
for image in os.lisrdir(path):
    print(image)
    img1 = cv2.imread(path + image)
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    #Call the feature extraction function
    x = feature_extraction(img)
    result = loaded_model.predict(x)
    segmented = result.reshape((img.reshape))
    
    
    
    
    

    

    
    
    
    
    
    
    
    
    
    








                    
                    
                    
                    
                    
                    
                    
                    
                    
    
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 00:38:16 2021

@author: abc
"""
"""
FEATURE BASED SEGMENTATION USING RANDOM FOREST
Demonstration using multiple training images

 STEP : 1 READ TRAINING IMAGES AND EXTRACT FEATURES
 
 STEP : 2 READ LABELED IMAGES (MASKS) AND CREATE ANOTHER DATAFRAME
  
 STEP : 3 GET DATA READY FOR RANDOM FOREST ( OR THE CLASSIFIER)
 
 STEP : 4 DEFINE THE CLASSIFIER AND FIT THE MODEL USING TRAINING DATA

 STEP : 5 CHECK ACCURACY OF THE MODEL

 STEP : 6 SAVE MODEL FOR FUTURE USE

 STEP : 7 MAKE PREDICTION ON NEW IMAGES  

"""

#import library's
import numpy as np
import cv2
import pandas as pd

import glob
import pickle
from matplotlib import pyplot as plt
import os

#################################################################

#STEP : 1 READ TRAINING IMAGES AND EXTRACT FEACTURES

#create dataframe to capture image features
image_dataset = pd.DataFrame()

#put the path of our train images
img_path = "C:\\Users\\abc\\Desktop\\Digital Sreeni\\Image Segmentation using Traditional machine learning\\train_images"

#iterate through each file
for image in os.listdir(img_path):
    print(image)
    
    #create temporary dataframe to capture information for each loop.
    df = pd.DataFrame()
    #Reset dataframe to blank after each loop
    
    #Read Images
    input_img = cv2.imread(img_path + image)
    print(input_img)
    
    #Check if the input image is RGB or GRAY and convert to GRAY if RGB
    if (len(input_img.shape)<3):
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    elif (len(input_img.shape) == 2) :
        img = input_img
    else :
        raise Exception('The module works only with gray scale and RGB image!!')
        
####################################################################################

#START ADDING DATA TO THE DATAFRAME 
 
    #Add pixel values to the dataframe
    pixel_values = img.reshape(-1)  #reshape our images
    df['Pixel_Value'] = pixel_values   #Pixel value it self as a feature
    df['Image_Name'] = image   #Capture image as we read multiple images
    
#####################################################################################

#GENERATE GABOR FEATURES

    num = 1  #To count the nuumbers up in order to Gabor features a label in the data frame
    kernels = []
    for theta in range(2):   #define number of thetas
        
        theta = theta / 4. * np.pi
    for sigma in (1,3): #sigma with 1 and 3
        for lamda in np.arange(0, np.pi, np.pi / 4):  #Range of wavelengths
            
            for gamma in (0.05,0.5):     #Gamma values of 0.05 and 0.5
               
                gabor_label = 'Gabor' + str(num) #Label Gabor columns as Gabor1, Gabor2 etc.
                #print(gabor_label)
                ksize = 9 #kernel size
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0 ,ktype = cv2.CV_32F)
                kernels.append(kernel)
               
                #Now filter the image and add values to a new column
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img   #Labels columns as Gabor1, Gabor2 , etc.
                print(gabor_label, ': theta = ',theta, ':sigma = ', sigma, ':lamda = ', lamda, ':gamma = ', gamma)
                num += 1  #Increment for gabor column label
            
####################################################################################

# GENERATE OTHER FEATURES AND ADD THEM TO THE DATA FRAME

    #Canny Edge
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe

    from skimage.filters import roberts, sobel, scharr, prewitt
    
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


#####################################################################################

#Update dataframe for images to include details for each image in the loop
    image_dataset = image_dataset.append(df)
    
    
######################################################################################

#STEP : 2 READ LABELED IMAGES (MASKS) AND CREATE ANOTHER DATAFRAME

#WITH LALBEL VALUES AND LABEL FILE NAMES

#create dataframe to capture mask info.
mask_dataset = pd.DataFrame()

mask_path = "C:\\Users\abc\Desktop\Digital Sreeni\Image Segmentation using Traditional machine learning\train_masks"

#iterate through each file to perform some action
for mask in os.listdir(mask_path):
    print(mask)
    
    #Create tamporary dataframe to capture ingo for each mask in the loop
    df2 = pd.DataFrame()
    #read the images
    input_mask = cv2.imread(mask_path + mask)
    
    #Check if the input mask is RGB or GRAY and convert to gray if RGB
    if input_mask.ndim == 3 and input_mask.shape[-1] == 3:
        label = cv2.cvtColor(input_mask, cv2.COLOR_BGR2GRAY)
    elif input_mask.ndim == 2:
        label = input_mask
    else:
        raise Exception("The module works only with grayscale and RGB images !!!")
        
    #Add pixel values to the data frame
    label_values = label.reshape(-1)
    df2['Label_Value'] = label_values
    df2['Mask_Name'] = mask
 
    #Update mask dataframe with all the info from each mask
    mask_dataset = mask_dataset.append(df2)
    
    
########################################################################################

#STEP : 3 GET DATA READY FOR RANDOM FOREST (OR ANY OTHER CLASSIFIER)

#COMBINE BOTH DATAFRAMES INTO A SINGLE DATASET

#Concatenate both image and mask datasets
dataset = pd.concat([image_dataset, mask_dataset], axis=1)

#if we do not want to include pixels with value 0
# e.g Sometimes unlabeled pixels may be given a value 0
dataset = dataset[dataset.Label_Value != 0]

#Assign training features to X and labels to Y
#Drop columns that are not relevant for training (non - features)
X = dataset.drop(labels = ["Image_Name", "Mask_Name", " Label_Value"], axis = 1)

#Assign label values to Y (our prediction)
Y = dataset["Label_Value"].values

#Split data into train and test to verify accuracy after fitting the model.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

#here test_size = 0.2 means we are looing forward our 20% data for testing and remaining data for training


#############################################################################################


#STEP : 4 DEFINE THE CLASSIFIER AND FIT A MODEL WITH OUR TRAINING DATA 

#Import training classifier 
from sklearn.ensemble import RandomForestClassifier

#Instantiate model with n numbers of decision trees 
model = RandomForestClassifier(n_estimators = 50, random_state= 42)

#Fit and train the model on training data 
model.fit(X_train, y_train)


##########################################################################################

#STEP : 5 ACCURACY CHECK 

from sklearn import metrics

#predict our test data
prediction_test = model.predict(X_test)

#Check Accuracy on test dataset 
print ("Accuracy =", metrics.accuracy_score(y_test, prediction_test))

#######################################################################################

#STEP : 6 SAVE MODEL FOR FUTURE USE

#WE can store the model for future use. In fact, this is how we do machine learning.
#Train on training images , validate on test images and deploy the model on unknown images.

#Save the trained modell as pickle string to disk for future use

#give the name of the model
model_name = "sandstone_model"

#dump our model into write and binary mode using pickle library
pickle.dump(model, open(model_name, 'wb'))

#To test the model on future datasets
#so firt of all load our model and dump into read and binary mode



 














 





               

               
   

    























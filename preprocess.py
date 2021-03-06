# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:15:27 2021

@author: ASUS
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import cv2


data_dir  = os.path.join(os.getcwd(),'clean_data')    #os.path helps in find the path of given directory
# getcwd( ) it is a function that help us determine current working directory
img_dir = os.path.join(os.getcwd(),'image')

def preprocess(image):           
    image = cv2.resize(image,(100,100))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return image
images = []
labels = []

for i in os.listdir(img_dir):              #here it os will give a list of images
    image = cv2.imread(os.path.join(img_dir,i))
    image = preprocess(image)
    images.append(image)
    labels.append(i.split('_')[0])

images = np.array(images)
labels = np.array(labels)


with open(os.path.join(data_dir,'images.p'),'wb') as f:
    pickle.dump(image,f)


with open(os.path.join(data_dir,'labels.p'),'wb') as f:
    pickle.dump(labels,f)
    
    
    
    
    
    
    
    
    
    
  #  to get video from webcam
  
import cv2
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0)

ret,frame = cap.read()

cap.release()
plt.imshow(frame)
plt.show()

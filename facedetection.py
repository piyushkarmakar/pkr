# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:06:14 2021

@author: cttc1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import urllib
URL="http://192.168.43.232:8080/shot.jpg"
face_data = "haarcascade_frontalface_default.xml"
classifier= cv2.CascadeClassifier(face_data) # this classifier is a model.fit kind of thing
data=[]
ret = True
while ret:
    img_url = urllib.request.urlopen(URL)
    image = np.array(bytearray(img_url.read()),np.uint8)
    frame = cv2.imdecode(image,-1)
    
    faces = classifier.detectMultiScale(frame,1.5,5)
    if faces is not None:
        for x,y,w,h in faces:
            face_image = frame[y:y+h,x:x+w].copy()
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
            if len(data)<=100:
                data.append(face_image)
            else: 
                cv2.putText(frame,'complete',(200,200),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
                
    cv2.imshow('capture',frame)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()
name = input("enter name :")
c = 0
for i in data:
    cv2.imwrite("images/"+name+'_'+str(c)+'.jpg',i)
    c+=1
'''for i in range(0,17):
    plt.imshow(data[i])
    plt.show()'''
#!/usr/bin/env python
# coding: utf-8

# # LIBRAIRIES

# In[1]:


import pandas as pd
import os
import glob
import cv2
import numpy as np


# # Variable à renseigner

# # --------------------------------------------------------------------------------------

# In[2]:


path_datastet="/Users/edouardnadaud/Desktop/DATACAMP_PYTHON_2/Dataset/trainset/" #localisation du dataset
dataset_original_name="trainset.csv"#nom du dataset.csv
file_name_new_dataset="datasetcrop" #Enter a name
destination_dataset="/Users/edouardnadaud/Desktop/DATACAMP_PYTHON_2/"  # Localisation du dossier ou creer le nouveau dataset


# # --------------------------------------------------------------------------------------

# chargement du dataset

# In[3]:


dataset_original_location=path_datastet+dataset_original_name
data=pd.read_csv(dataset_original_location) 


# Création du chemin de destination du nouveaux dataset

# In[4]:


path_new_dataset=destination_dataset+file_name_new_dataset+"/"


# creation du dossier datasetcrop

# In[5]:


if not os.path.exists(path_new_dataset):
        os.makedirs(path_new_dataset)


# Récuperation du nom et des point nécesssaire pour aligner

# In[6]:


filename=data[['filename','label',' x_39',' x_45',' y_39', ' y_45']]


# calcul du nombre de file du dataset

# In[7]:


type_file_dataset_original = (path_datastet + "*.png")
nbr_file=len(glob.glob(type_file_dataset_original))


# ## Crop

# In[8]:


index=0
for i in range(nbr_file):
    
    
    path_dst=destination_dataset+file_name_new_dataset+"/"
    
    file_src=path_datastet+str(filename['filename'][i])+".png"
    file=str(filename['filename'][i])
    file_dst=path_dst+str(filename['filename'][i])+"crop.png"
    img = cv2.imread(file_src)

    


    #detection facial
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    (x, y, w, h) = faces_detected[0]
    H, W = img.shape[:2]

    #pour les yeux je choisis les points 39 et 45
    
    img_row=data.loc[data['filename']==file,:]
    
    #recuperation des coordonnée

        #x39
    Xleft_eye=img_row[" x_39"]
    Xleft_eye=Xleft_eye[index]
    
        #y39
    Yleft_eye=img_row[" y_39"]
    Yleft_eye=Yleft_eye[index]
        #x45
    Xright_eye=img_row[" x_45"]
    Xright_eye=Xright_eye[index]
    
        #y45
    Yright_eye=img_row[" y_45"]
    Yright_eye=Yright_eye[index]

    #calculation de l'angle
    delta_x = Xright_eye - Xleft_eye
    delta_y = Yright_eye - Yleft_eye
    angle=np.arctan(delta_y/delta_x)
    angle = (angle * 180) / np.pi


    # Calculating a center point of the image
    # Integer division "//"" ensures that we receive whole numbers
    center = (W // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, (angle), 1.0)    

    rotated = cv2.warpAffine(img, M, (W, H))


    #On centre l'image sur le visage
    p=0
    cv2.imwrite(file_dst, rotated[y-p+1:y+h+p, x-p+1:x+w+p])

    index=index+1


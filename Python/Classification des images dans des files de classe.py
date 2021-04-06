#!/usr/bin/env python
# coding: utf-8

# # Librairies

# In[1]:


import pandas as pd
import numpy as np
import os
import glob
import shutil


# # Variables à renseigner

# # --------------------------------------------------------------------------------------

# In[2]:


file_name="CNN_Classe" #Nom du file ou seront classé les données
path_file="/Users/edouardnadaud/Desktop/DATACAMP_PYTHON_2/datasetcrop/" #chemin ou sera le fichier 
path_dst=path_file+file_name+"/"
dataset_original_name="trainset.csv"#nom du dataset.csv
path_datastet="/Users/edouardnadaud/Desktop/DATACAMP_PYTHON_2/Dataset/trainset/" #localisation du dataset


# # --------------------------------------------------------------------------------------

# Chargement du dataset

# In[3]:


dataset_original_location=path_datastet+dataset_original_name
data=pd.read_csv(dataset_original_location) 
label=data["label"]
labels_names=np.unique(label)
filename=data[['filename','label',' x_39',' x_45',' y_39', ' y_45']]


# Création des dossiers

# In[4]:


if not os.path.exists(path_dst):
        os.makedirs(path_dst)
for i in labels_names:
    if not os.path.exists(path_dst+str(i)):
        os.makedirs(path_dst+str(i))


# Nombre de fichier a classer

# In[5]:


type_file_dataset_original = (path_datastet + "*.png")
nbr_file=len(glob.glob(type_file_dataset_original))


# copie des images dans leur dossier de classe

# In[6]:


for i in range(nbr_file):
    path_src=path_file
    path_dest=path_dst+"/"
    
    file_src=path_src+str(filename['filename'][i])+"crop.png"
    
    
    for classe in labels_names: 
        if ((filename["label"][i])== classe):
            
            file_dst=path_dest+str(classe)+"/"+str(filename['filename'][i])+"crop.png"
            
            shutil.copyfile(file_src,file_dst)



# In[ ]:





# In[ ]:





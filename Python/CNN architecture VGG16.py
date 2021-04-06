#!/usr/bin/env python
# coding: utf-8

# # LIBRAIRIE

# In[1]:


import pandas as pd
import numpy as np
import os
import glob
import shutil
import cv2
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv("/Users/edouardnadaud/Desktop/DATACAMP_PYTHON_2/Dataset/trainset/trainset.csv") 


# In[3]:


a=np.unique(data['label'],return_counts=True)


# In[4]:


def Proportion_class(data):
    #librairie de la fonction
#--------------------------------------------------------------------------------------------------------#
    import pandas as pd
    import numpy as np
#--------------------------------------------------------------------------------------------------------#
    #Récuperation des noms et labels associé aux images
    data=data[['filename','label']]
    
    #Proportions des classes
    nbr_class=np.unique(data['label'],return_counts=True)
    
    #nombres de données
    nbr_données=np.sum(nbr_class[1])
        
    #Proportions de chaque classe
    prop_class=[]
    for i in range(len(nbr_class[1])):
        prop_class.append(nbr_class[1][i]/nbr_données)
    
    return(prop_class)
    
    
        
   


# In[5]:


train_data_path = "/Users/edouardnadaud/Desktop/DATACAMP_PYTHON_2/datasetcrop/CNN_Classe"
val_data_path = "/Users/edouardnadaud/Desktop/DATACAMP_PYTHON_2/datasetcrop/CNN_CLASS_VAL"


# In[6]:



batch_size=3
img_height = 224
img_width = 224
num_classes = 8


train_data = tf.keras.preprocessing.image_dataset_from_directory(
  train_data_path,
  validation_split=0.2,
  subset="training",
  seed=42,
    #je redimensionne les images
  image_size=(img_height, img_width),
  batch_size=batch_size,
  )

val_data = tf.keras.preprocessing.image_dataset_from_directory(
  train_data_path,

  validation_split=0.2,
  subset="validation",
  seed=42,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = val_data.class_names
print(class_names)


# In[8]:


plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
    for i in range(3):
        ax = plt.subplot(1, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# In[13]:


#Import Libraries

import sys
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D,MaxPooling2D, Activation, Dropout, BatchNormalization, Input
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import itertools  

############################################
# Set the paths for training, testing and validation 
#train_data_path = "/Users/edouardnadaud/Desktop/DATACAMP_PYTHON_2/datasetcrop/CNN_Classe"
#val_data_path = "/Users/edouardnadaud/Desktop/DATACAMP_PYTHON_2/datasetcrop/CNN_CLASS_VAL"
path_train = "/Users/edouardnadaud/Desktop/DATACAMP_PYTHON_2/datasetcrop/CNN_Classe"
path_val= "/Users/edouardnadaud/Desktop/DATACAMP_PYTHON_2/datasetcrop/CNN_Classe_val"
path_test= "/Users/edouardnadaud/Desktop/DATACAMP_PYTHON_2/datasetcrop/CNN_Classe_test"
train_data_path = path
test_data_path = path
valid_data_path =path

############################################
# Set image size and batch size

img_rows = 224
img_cols = 224
batch_size = 3



############################################
# Set Data Generator for training, testing and validataion.
# Note for testing, set shuffle = false (For proper Confusion matrix)

train_datagen = ImageDataGenerator(zoom_range=0.5)
train_generator = train_datagen.flow_from_directory(train_data_path,
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    class_mode='categorical', shuffle=True)

valid_datagen = ImageDataGenerator()
valid_generator = valid_datagen.flow_from_directory(valid_data_path,
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    class_mode='categorical', shuffle=True)

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(test_data_path,
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    class_mode='categorical', shuffle=False)

#########################################################################
# Function for plots images with labels within jupyter notebook

def plots(ims, figsize=(12,12), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1

    for i in range(len(ims)):
        sp = f.add_subplot(cols, rows, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=12)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

#########################################################################

#Check the training set (with batch of 10 as defined above
imgs, labels = next(train_generator)

#Images are shown in the output
plots(imgs, titles=labels)

#Images Classes with index
print(train_generator.class_indices)

#Model Creation / Sequential
model = Sequential([Conv2D(input_shape=(img_rows,img_cols,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"),
                    Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"),
                    MaxPooling2D(pool_size=(2,2),strides=(2,2)),
                    Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
                    Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"), 
                    MaxPool2D(pool_size=(2,2),strides=(2,2)),
                    Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"), 
                    Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
                    Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
                    Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
                    MaxPool2D(pool_size=(2,2),strides=(2,2)),
                    Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                    Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                    Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                    MaxPool2D(pool_size=(2,2),strides=(2,2)),
                    Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                    Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                    Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                    MaxPool2D(pool_size=(2,2),strides=(2,2)),                    

                    Flatten(), 
                    Dense(7, activation='softmax')
                   ])

#Get summary of the model
model.summary()

#Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


class_weight={
    0:0.34,
    1:1.,
    2:0.93,
    3:0.97,
    4:0.97,
    5:0.92,
    6:0.96,
    7:0.9
    
}



#Train the model
history = model.fit(train_generator,
                    steps_per_epoch=50, 
                    validation_data=valid_generator, 
                    validation_steps=17,
                    epochs=25,
                    class_weight=class_weight    
                   )


#Get the accuracy score
test_score = model.evaluate_generator(test_generator, batch_size)

print("[INFO] accuracy: {:.2f}%".format(test_score[1] * 100)) 
print("[INFO] Loss: ",test_score[0])

#Plot the Graph

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
  
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['accuracy'],'r',linewidth=3.0)
plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)


##################################################################################################
#Plot the confusion matrix. Set Normalize = True/False

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10,10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

##################################################################################################


#Print the Target names

target_names = []
for key in train_generator.class_indices:
    target_names.append(key)

# print(target_names)

#Confution Matrix 

Y_pred = model.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(test_generator.classes, y_pred)
plot_confusion_matrix(cm, target_names, title='Confusion Matrix')

#Print Classification Report
print('Classification Report')
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

#Save the model
model.save("tutorial.hdf5")



model = load_model('tutorial.hdf5')

file = 'sample.jpg'

img = cv.cvtColor(cv.imread(file),cv.COLOR_BGR2RGB)
img = cv.resize(img, (img_rows,img_cols))

test_image = image.img_to_array(img)
test_image = np.expand_dims(test_image, axis=0)
pred = model.predict(test_image)
print(pred, labels[np.argmax(pred)])




# In[ ]:





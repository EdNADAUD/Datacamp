#!/usr/bin/env python
# coding: utf-8

# In[3]:


def imbalenced_to_balanced_dataset(dataset):
    #librairie de la fonction
#--------------------------------------------------------------------------------------------------------#
    import imblearn
    from imblearn.over_sampling import SMOTE
    import pandas as pd
    from collections import Counter
#--------------------------------------------------------------------------------------------------------#
    data=dataset
    y=data["label"]
    del data['label']
    del data['filename']
    X=data
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    counter = Counter(y)
    print(counter)
    return(X,y)


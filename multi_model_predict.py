#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 01:09:07 2020

@author: ganja
"""

#%%########
# Imports #
###########

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop

from sklearn.externals.joblib import load
 
from sklearn.metrics import confusion_matrix

from preprocessing import preprocess_data

#%%###################
# Data Preprocessing #
######################

# Importing the dataset
dataset = pd.read_csv('train.csv')

# Split the dataset
Y = dataset.iloc[:,1].values
dataset = dataset.drop('Survived', axis=1)

dataset = np.array(preprocess_data(dataset))

#%%###############
# Loading Models #
##################

models = []

for i in range(10):
    print("loading model", i)
    models.append(keras.models.load_model("trained_models/model_%d.h5" % (i)))

#%%##################
# Predicting Values #
#####################

def predict(dataset):
    preds = []
    
    for i in range(len(dataset)):
        tmp_pred = np.ndarray((10,))
        for j in range(len(models)):
            tmp_arr = np.ndarray((1, 35))
            tmp_arr[0] = dataset[i]
            tmp_pred[j] = models[j].predict(tmp_arr)
        tmp_pred.sort(axis=0)
        preds.append(np.mean(tmp_pred))
    
    preds = np.array(preds).round()
    return preds

preds = predict(dataset)

cm = confusion_matrix(Y, np.array(preds).round())
print('accuracy:',(cm[0,0]+cm[1,1])/len(Y))

#%%#######
# Kaggle #
##########

test_dataset = pd.read_csv('test.csv')

ids = test_dataset.iloc[:,0]

test_dataset = preprocess_data(test_dataset)

test_preds = predict(test_dataset.values)

test_preds = np.array(test_preds, dtype=int)

submit = pd.DataFrame({'PassengerId': ids, 'Survived': test_preds})
submit.to_csv('submit.csv', index=False)
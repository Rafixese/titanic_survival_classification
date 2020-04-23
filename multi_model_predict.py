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

#%%###################
# Data Preprocessing #
######################

# Importing the dataset
dataset = pd.read_csv('train.csv')

# Split the dataset
Y = dataset.iloc[:,1].values
dataset = dataset.drop('Survived', axis=1)

def preprocess_data(dataset):

    # Dropping unnecessary columns
    dataset = dataset.drop(['PassengerId', 'Ticket', 'Cabin'], axis = 1)
    
    # Picking out title from name column
    
    dataset['Name'] = [name.split(',')[1].strip().split('.')[0] for name in dataset['Name']]
    
    # Check for nan(anananananananananana Batman!)
    
    dataset.isna().any() # cols with nan(ana Batman!): ['Age', 'Embarked']
    
    dataset['Age'] = dataset['Age'].fillna(int(dataset['Age'].dropna().median()))
    
    dataset['Embarked'].describe() # top: S
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
    # dataset.isna().any() # We are good!
    
    # Encoding the dataset
    title_label_encoder = LabelEncoder()
    title_label_encoder.classes_ = np.load('fitted_data_processors/title_label_encoder.npy', allow_pickle = True)
    
    for i in range(len(dataset['Name'])):
        if dataset['Name'][i] not in title_label_encoder.classes_: dataset['Name'][i] = 'Mr'
    
    dataset['Name'] = title_label_encoder.transform(dataset['Name'])
    # print(title_label_encoder.classes_)
    sex_label_encoder = LabelEncoder()
    sex_label_encoder.classes_ = np.load('fitted_data_processors/sex_label_encoder.npy', allow_pickle = True)
    dataset['Sex'] = sex_label_encoder.transform(dataset['Sex'])
    # print(sex_label_encoder.classes_)
    embarked_label_encoder = LabelEncoder()
    embarked_label_encoder.classes_ = np.load('fitted_data_processors/embarked_label_encoder.npy', allow_pickle = True)
    dataset['Embarked'] = embarked_label_encoder.transform(dataset['Embarked'])
    # print(embarked_label_encoder.classes_)
    one_hot_encoder = OneHotEncoder( handle_unknown = "ignore" )
    one_hot_encoder.categories_ = np.load('fitted_data_processors/one_hot_encoder.npy', allow_pickle = True)
    
    dataset_ohe = dataset.iloc[:,[0, 1, 4, 5, 7]].values
    dataset_ohe = one_hot_encoder.transform(dataset_ohe).toarray()
    
    col_to_drop = [0]
    for i in range(len(one_hot_encoder.categories_)-1):
        col_to_drop.append(col_to_drop[i] + len(one_hot_encoder.categories_[i]))
        
    dataset_ohe = np.delete(dataset_ohe, col_to_drop, axis=1)
    
    dataset = dataset.drop(['Pclass', 'Name', 'SibSp', 'Parch', 'Embarked'], axis=1)
    dataset = dataset.join(pd.DataFrame(dataset_ohe))
    
    # dataset = pd.get_dummies(dataset, columns = ['Pclass', 'Name', 'SibSp', 'Parch', 'Embarked'], drop_first = True)
    
    # Scaling the dataset
    
    age_scaler = load('fitted_data_processors/age_scaler.bin')
    dataset['Age'] = age_scaler.fit_transform(dataset['Age'].values.reshape(-1,1))
    
    fare_scaler = load('fitted_data_processors/fare_scaler.bin')
    dataset['Fare'] = fare_scaler.fit_transform(dataset['Fare'].values.reshape(-1,1))
    
    return dataset

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

submit = pd.DataFrame({'PassengerId': ids, 'Survived': test_preds})
submit.to_csv('submit.csv', index=False)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:25:19 2020

@author: ganja
"""

#%%########
# Imports #
###########

import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

#%%###################
# Data Preprocessing #
######################

# Importing the dataset
dataset = pd.read_csv('train.csv')

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
dataset['Name'] = title_label_encoder.fit_transform(dataset['Name'])
print(title_label_encoder.classes_)

sex_label_encoder = LabelEncoder()
dataset['Sex'] = sex_label_encoder.fit_transform(dataset['Sex'])
print(sex_label_encoder.classes_)

embarked_label_encoder = LabelEncoder()
dataset['Embarked'] = embarked_label_encoder.fit_transform(dataset['Embarked'])
print(embarked_label_encoder.classes_)

dataset = pd.get_dummies(dataset, columns = ['Pclass', 'Name', 'SibSp', 'Parch', 'Embarked'], drop_first = True)

# Scaling the dataset

age_scaler = StandardScaler()
dataset['Age'] = age_scaler.fit_transform(dataset['Age'].values.reshape(-1,1))

fare_scaler = StandardScaler()
dataset['Fare'] = fare_scaler.fit_transform(dataset['Fare'].values.reshape(-1,1))

# Split the dataset

X, Y = dataset.iloc[:,1:].values, dataset.iloc[:,0].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.15, random_state = 0)


#%%###############
# Building model #
##################

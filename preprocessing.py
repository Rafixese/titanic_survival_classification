#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 16:24:49 2020

@author: ganja
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.externals.joblib import load

def preprocess_data(X):

    # Dropping unnecessary columns
    X = X.drop(['PassengerId', 'Ticket', 'Cabin'], axis = 1)
    
    # Picking out title from name column
    
    X['Name'] = [name.split(',')[1].strip().split('.')[0] for name in X['Name']]
    
    # Check for nan(anananananananananana Batman!)
    
    X.isna().any() # cols with nan(ana Batman!): ['Age', 'Embarked']
    
    X['Age'] = X['Age'].fillna(int(X['Age'].dropna().median()))
    
    X['Embarked'].describe() # top: S
    X['Embarked'] = X['Embarked'].fillna('S')
    
    # X.isna().any() # We are good!
    
    # Encoding the X
    title_label_encoder = LabelEncoder()
    title_label_encoder.classes_ = np.load('fitted_data_processors/title_label_encoder.npy', allow_pickle = True)
    
    for i in range(len(X['Name'])):
        if X['Name'][i] not in title_label_encoder.classes_: X['Name'][i] = 'Mr'
    
    X['Name'] = title_label_encoder.transform(X['Name'])
    # print(title_label_encoder.classes_)
    sex_label_encoder = LabelEncoder()
    sex_label_encoder.classes_ = np.load('fitted_data_processors/sex_label_encoder.npy', allow_pickle = True)
    X['Sex'] = sex_label_encoder.transform(X['Sex'])
    # print(sex_label_encoder.classes_)
    embarked_label_encoder = LabelEncoder()
    embarked_label_encoder.classes_ = np.load('fitted_data_processors/embarked_label_encoder.npy', allow_pickle = True)
    X['Embarked'] = embarked_label_encoder.transform(X['Embarked'])
    # print(embarked_label_encoder.classes_)
    one_hot_encoder = OneHotEncoder( handle_unknown = "ignore" )
    one_hot_encoder.categories_ = np.load('fitted_data_processors/one_hot_encoder.npy', allow_pickle = True)
    
    X_ohe = X.iloc[:,[0, 1, 4, 5, 7]].values
    X_ohe = one_hot_encoder.transform(X_ohe).toarray()
    
    col_to_drop = [0]
    for i in range(len(one_hot_encoder.categories_)-1):
        col_to_drop.append(col_to_drop[i] + len(one_hot_encoder.categories_[i]))
        
    X_ohe = np.delete(X_ohe, col_to_drop, axis=1)
    
    X = X.drop(['Pclass', 'Name', 'SibSp', 'Parch', 'Embarked'], axis=1)
    X = X.join(pd.DataFrame(X_ohe))
    
    # X = pd.get_dummies(X, columns = ['Pclass', 'Name', 'SibSp', 'Parch', 'Embarked'], drop_first = True)
    
    # Scaling the X
    
    age_scaler = load('fitted_data_processors/age_scaler.bin')
    X['Age'] = age_scaler.fit_transform(X['Age'].values.reshape(-1,1))
    
    fare_scaler = load('fitted_data_processors/fare_scaler.bin')
    X['Fare'] = fare_scaler.fit_transform(X['Fare'].values.reshape(-1,1))
    
    # Save data processing fitted scalers and encoders
    
    # np.save('fitted_data_processors/title_label_encoder.npy',title_label_encoder.classes_)
    # np.save('fitted_data_processors/sex_label_encoder.npy',sex_label_encoder.classes_)
    # np.save('fitted_data_processors/embarked_label_encoder.npy',embarked_label_encoder.classes_)
    
    # np.save('fitted_data_processors/one_hot_encoder.npy',one_hot_encoder.categories_)
    
    # dump(age_scaler, 'fitted_data_processors/age_scaler.bin', compress=True)
    # dump(fare_scaler, 'fitted_data_processors/fare_scaler.bin', compress=True)
    
    return X
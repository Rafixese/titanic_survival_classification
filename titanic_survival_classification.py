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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop

# from sklearn.externals.joblib import dump

from preprocessing import preprocess_data

#%%###################
# Data Preprocessing #
######################

# Importing the dataset
dataset = pd.read_csv('train.csv')

Y =  dataset.iloc[:,1].values
dataset = dataset.drop('Survived', axis=1)

# def preprocess_data(dataset):

#     # Dropping unnecessary columns
#     dataset = dataset.drop(['PassengerId', 'Ticket', 'Cabin'], axis = 1)
    
#     # Picking out title from name column
    
#     dataset['Name'] = [name.split(',')[1].strip().split('.')[0] for name in dataset['Name']]
    
#     # Check for nan(anananananananananana Batman!)
    
#     dataset.isna().any() # cols with nan(ana Batman!): ['Age', 'Embarked']
    
#     dataset['Age'] = dataset['Age'].fillna(int(dataset['Age'].dropna().median()))
    
#     dataset['Embarked'].describe() # top: S
#     dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
#     # dataset.isna().any() # We are good!
    
#     # Encoding the dataset
    
#     title_label_encoder = LabelEncoder()
#     dataset['Name'] = title_label_encoder.fit_transform(dataset['Name'])
#     # print(title_label_encoder.classes_)
    
#     sex_label_encoder = LabelEncoder()
#     dataset['Sex'] = sex_label_encoder.fit_transform(dataset['Sex'])
#     # print(sex_label_encoder.classes_)
    
#     embarked_label_encoder = LabelEncoder()
#     dataset['Embarked'] = embarked_label_encoder.fit_transform(dataset['Embarked'])
#     # print(embarked_label_encoder.classes_)
    
#     one_hot_encoder = OneHotEncoder()
    
#     dataset_ohe = dataset.iloc[:,[1, 2, 5, 6, 8]].values
#     one_hot_encoder.fit(dataset_ohe)
#     one_hot_encoder.categories_
#     dataset_ohe = one_hot_encoder.transform(dataset_ohe).toarray()
    
#     col_to_drop = [0]
#     for i in range(len(one_hot_encoder.categories_)-1):
#         col_to_drop.append(col_to_drop[i] + len(one_hot_encoder.categories_[i]))
        
#     dataset_ohe = np.delete(dataset_ohe, col_to_drop, axis=1)
    
#     dataset = dataset.drop(['Pclass', 'Name', 'SibSp', 'Parch', 'Embarked'], axis=1)
#     dataset = dataset.join(pd.DataFrame(dataset_ohe))
    
#     # dataset = pd.get_dummies(dataset, columns = ['Pclass', 'Name', 'SibSp', 'Parch', 'Embarked'], drop_first = True)
    
#     # Scaling the dataset
    
#     age_scaler = StandardScaler()
#     dataset['Age'] = age_scaler.fit_transform(dataset['Age'].values.reshape(-1,1))
    
#     fare_scaler = StandardScaler()
#     dataset['Fare'] = fare_scaler.fit_transform(dataset['Fare'].values.reshape(-1,1))
    
#     # Save data processing fitted scalers and encoders
    
#     # np.save('fitted_data_processors/title_label_encoder.npy',title_label_encoder.classes_)
#     # np.save('fitted_data_processors/sex_label_encoder.npy',sex_label_encoder.classes_)
#     # np.save('fitted_data_processors/embarked_label_encoder.npy',embarked_label_encoder.classes_)
    
#     # np.save('fitted_data_processors/one_hot_encoder.npy',one_hot_encoder.categories_)
    
#     # dump(age_scaler, 'fitted_data_processors/age_scaler.bin', compress=True)
#     # dump(fare_scaler, 'fitted_data_processors/fare_scaler.bin', compress=True)
#     return dataset

X = preprocess_data(dataset)

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.15, random_state = 0)

#%%###############
# Building model #
##################

def build_model(optimizer):

    model = Sequential()
    
    model.add(Dense(64, activation = 'relu', input_shape=(35,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(64, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(32, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(32, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.summary()
    
    model.compile(optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
    
    return model
    
#%%################
# Finding best lr #
###################

lr_acc = []

for i in range(1,100):
    temp_opt = RMSprop(lr= i / 10000.0)
    temp_model = build_model(temp_opt)
    history = temp_model.fit(X_train, Y_train, batch_size = 16, epochs = 30, validation_data = (X_test, Y_test))
    acc = history.history['val_accuracy']
    lr_acc.append(max(acc))

#%%#####################
# Final training model #
########################

optimizer = RMSprop(lr= 0.001)
model = build_model('rmsprop')

history = model.fit(X_train, Y_train, batch_size = 16, epochs = 11, validation_data = (X_test, Y_test))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label = 'Train Accuracy')
plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'b', label = 'Train Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%####################
# Save 10 best models #
#######################

for i in range(10):
    models = []
    model_accs = []
    for y in range(10):
        optimizer = RMSprop(lr= 0.001)
        model = build_model('rmsprop')
        history = model.fit(X_train, Y_train, batch_size = 16, epochs = 11, validation_data = (X_test, Y_test))
        models.append(model)
        model_accs.append(history.history['val_accuracy'][-1])
    
    best_model_index = model_accs.index(max(model_accs))
    models[best_model_index].save("model_%d.h5" % (i))
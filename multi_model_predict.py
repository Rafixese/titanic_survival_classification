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
import numpy as np
import keras

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from preprocessing import preprocess_data

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt

#%%###################
# Data Preprocessing #
######################

# Importing the dataset
dataset = pd.read_csv('train.csv')

# Split the dataset
Y = dataset.iloc[:,1].values
dataset = dataset.drop('Survived', axis=1)

X = np.array(preprocess_data(dataset))

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.15, random_state = 0)

#%%###############
# Loading Models #
##################

models = []

for i in range(10):
    print("loading model", i)
    models.append(keras.models.load_model("trained_models/model_%d.h5" % (i)))
  
#%%######################
# Predicts by 10 Models #
#########################

def predict_10_models(X):
    preds = []
    
    for i in range(len(X)):
        tmp_pred = np.ndarray((10,))
        for j in range(len(models)):
            tmp_arr = np.ndarray((1, 35))
            tmp_arr[0] = X[i]
            tmp_pred[j] = models[j].predict(tmp_arr)
        preds.append(tmp_pred)
    
    preds = np.array(preds)
    return np.array(preds)

train_model_preds = predict_10_models(X_train)
valid_model_preds = predict_10_models(X_test)
    
#%%####################################################################
# Creating new model to predict final outcome from 10 models predicts #
#######################################################################

def build_model(optimizer):

    model = Sequential()
    
    model.add(Dense(20, activation = 'relu', input_shape=(10,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(10, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.summary()
    
    model.compile(optimizer, loss='binary_crossentropy', metrics = ['accuracy'])
    
    return model


model = build_model('rmsprop')

history = model.fit(train_model_preds, Y_train, batch_size = 16, epochs = 8, validation_data = (valid_model_preds, Y_test))

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


#%%##################
# Predicting Values #
#####################

# def predict(X):
#     preds = []
    
#     for i in range(len(X)):
#         tmp_pred = np.ndarray((10,))
#         for j in range(len(models)):
#             tmp_arr = np.ndarray((1, 35))
#             tmp_arr[0] = X[i]
#             tmp_pred[j] = models[j].predict(tmp_arr)
#         tmp_pred.sort(axis=0)
#         preds.append(np.mean(tmp_pred))
    
#     preds = np.array(preds).round()
#     return preds

# preds = predict(X_test)

# cm = confusion_matrix(Y_test, np.array(preds).round())
# print('accuracy:',(cm[0,0]+cm[1,1])/len(Y_test))

#%%#######
# Kaggle #
##########

test_dataset = pd.read_csv('test.csv')

ids = test_dataset.iloc[:,0]

test_dataset = preprocess_data(test_dataset)

test_preds_10_models = predict_10_models(test_dataset.values)

test_preds = model.predict(test_preds_10_models).round()

test_preds = np.array(test_preds, dtype=int).reshape((test_preds.shape[0],))

submit = pd.DataFrame({'PassengerId': ids, 'Survived': test_preds})
submit.to_csv('submit.csv', index=False)
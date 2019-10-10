''' 
Author : Shalom Lee Li-Ling
Date: 10/10/2019
Title: models/stacked_ensemble/stacked_ensemble.py
Code Version: 
Availability: 
''' 
'''
This file is extracted from Google Colab notebook stacked_ensemble.ipynb (neenyabigki6). 
It trains the Level 1 model with 10 training scenes, 5 validation and 5 testing scenes as described in the final report
'''
import numpy as np
import math
import tensorflow as tf
import random
import copy
import os
from matplotlib import pyplot 
%matplotlib inline
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from numpy import argmax
from livelossplot import PlotLossesKeras
import requests

# send message to telegram app on phone once training is done
def telegram(message):

    bot_token = '923260274:AAHyCqG6jpPI_xo0x2wJvhGVA_oeUnMf41Y'
    bot_chatID = '763781536'
    send = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + message
    try:
        response = requests.get(send)
        response.raise_for_status()
    except requests.exceptions.RequestException as err:
        print ("OOps: Something Else",err)
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)  
    return response.json()

# get all required train, test and validation data
def get_valid_data(mode): 
    in_folder = os.path.join("/content/drive/My Drive/Colab Notebooks/data")
    
    if (mode=="train"):
        x = np.load(os.path.join(in_folder, "X_train.npy"))
        y = np.load(os.path.join(in_folder, "Y_train.npy"))
        
    elif (mode=="test"):
        x = np.load(os.path.join(in_folder, "X_test.npy"))
        y = np.load(os.path.join(in_folder, "Y_test.npy"))
        
    else:
        x = np.load(os.path.join(in_folder, "X_valid.npy"))
        y = np.load(os.path.join(in_folder, "Y_valid.npy"))
    
    return x, y

# define train, test, validation ratio and split data - scene 30-40 for training, next 5 for validation and final 5 for testing
X, y = get_valid_data(mode="train")

X_train = X[:, 30:, :, :, :]
y_train = y[:, 30:, :]
X_valid, y_valid = get_valid_data(mode="valid")

# reshape into propor shape for CNN modelling
X1_train = X_train.reshape(-1, 46, 500, 1)
X1_valid = X_valid.reshape(-1, 46, 500, 1)
Y1_train = y_train.reshape(-1, 61)
Y1_valid = y_valid.reshape(-1, 61)

nrows = X1_train.shape[1]
ncolumns = X1_train.shape[2]

input_shape = (nrows, ncolumns, 1)

print(X1_train.shape[0], 'train samples')
print(X1_valid.shape[0], 'valid samples')
print("Finished reading datasets")

numClasses = y_train.shape[1]

# load level 0 models from file
def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        filename = '/content/drive/My Drive/Colab Notebooks/trial3_data/cnn_data_trial3data/model_' + str(i + 1) + '_short.h5'
       
        model = load_model(filename)        
        all_models.append(model)
        print('loaded %s' % filename)
        
    return all_models

# define stacked model from multiple member input models
def define_stacked_model(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for j,layer in enumerate(model.layers):
            # rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
            print(layer.name)
            
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:       
            # make not trainable
            layer.trainable = False
            
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    
    merge = concatenate(ensemble_outputs, axis=-1)
    hidden = Dense(128, activation='relu')(merge)
    second = Dropout(0.8)(hidden)
    output = Dense(61, activation='softmax')(second)

    model = Model(inputs=ensemble_visible, outputs=output)

    # plot graph of ensemble
    plot_model(model, show_shapes=True, to_file='/content/drive/My Drive/Colab Notebooks/stack/stack_model_graph5.png')
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fit a stacked model
def fit_stacked_model(model, inputX, inputy, validX, validy):
    
    X = [inputX for _ in range(len(model.input))]
    X_v = [validX for _ in range(len(model.input))]
    
    checkpoint_path = "/content/drive/My Drive/Colab Notebooks/stack/cpCombined5-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_loss',mode='min', verbose=1, save_best_only=True)
    
    csv_logger = CSVLogger('/content/drive/My Drive/Colab Notebooks/stack/training5.log', append=True, separator=',')
    model.load_weights('/content/drive/My Drive/Colab Notebooks/stack/cpCombined5-0030.ckpt')
    # fit model
    history = model.fit(X, inputy, epochs=200, initial_epoch = 30, verbose=1, validation_data=(X_v, Y1_valid), callbacks=[cp_callback, csv_logger, PlotLossesKeras()])
    return history

# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
     
    # make prediction
    return model.predict(X, verbose=0)

# load all models
n_members = 3
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# define ensemble model
stacked_model = define_stacked_model(members)

# fit stacked model on train dataset
history = fit_stacked_model(stacked_model, X1_train, Y1_train, X1_valid, Y1_train)

# message done to telegram app
telegram('stack cnn done')

# prediction 
inputX = X1_test
X = [inputX for _ in range(len(model.input))]     
test_loss, test_acc = model.evaluate(X, Y1_test, verbose = 0)

inputX = X1_valid
X = [inputX for _ in range(len(model.input))] 
valid_loss, valid_acc = model.evaluate(X, Y1_valid, verbose = 0)

inputX = X1_train
X = [inputX for _ in range(len(model.input))] 
train_loss, train_acc = model.evaluate(X, Y1_train, verbose = 0)
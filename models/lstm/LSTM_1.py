''' 
Author : Shalom Lee Li-Ling
Date: 10/10/2019
Title: models/lstm/LSTM_1.py
Code Version: 
Availability: 
''' 
'''
This file is extracted from Google Colab notebook LSTM_1.ipynb (neenyabigki4 and neenyabigki6). 
It trains LSTM1 model as described in the final report
'''
import numpy as np
import os
import csv
import re
import sys
import math
import random
import requests
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, TimeDistributed, Activation, Dense, Bidirectional
from tensorflow.keras.optimizers import SGD, Adam, Adadelta
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from livelossplot import PlotLossesKeras

# to send message to telegram app
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

# fit generator
def generate_array(mode="train"):
    while True:
      in_folder = os.path.join("/content","drive","My Drive","Colab Notebooks","lstm_data")
      reorder = 0

      if(mode=="train"):
        current_timestep = random.randint(0, 39)
        if current_timestep >=31:
          reorder = 1
        else:
          reorder = 0

      elif(mode=="valid"):
        current_timestep = random.randint(40, 45)
        
      elif(mode=="test"):
        current_timestep = random.randint(45, 49)
      
      x = np.load(os.path.join(in_folder, "in_step{}.npy".format(current_timestep)))
      y = np.load(os.path.join(in_folder, "out_step{}.npy".format(current_timestep)))

      if reorder:
        max_num = 40
        repeat_last = 10 - (max_num - current_timestep)

        X = x[:,0:(10-repeat_last),:]
        Y = y[:,0:(10-repeat_last),:]
        X_new = np.zeros((X.shape[0], 10, X.shape[2]))
        Y_new = np.zeros((Y.shape[0], 10, Y.shape[2]))

        for k in range(0, X.shape[0]):
            for i in range(0, X_new.shape[1]):
                if i < X.shape[1]:
                    X_new[k,i,:] = X[k,i,:]
                    Y_new[k,i,:] = Y[k,i,:]
                else:
                    X_new[k,i,:] = X_new[k,i-1,:]
                    Y_new[k,i,:] = Y_new[k,i-1,:]
        X = X_new
        Y = Y_new

      X, Y = shuffle(x, y)

      yield (X,Y)

# load all data
X_valid = np.load(os.path.join("/content","drive","My Drive","Colab Notebooks","lstm_data", "in_step40.npy"))
Y_valid = np.load(os.path.join("/content","drive","My Drive","Colab Notebooks","lstm_data", "out_step40.npy"))
X_valid = X_valid[:,0:5]
Y_valid = Y_valid[:,0:5]
X_test = np.load(os.path.join("/content","drive","My Drive","Colab Notebooks","lstm_data", "in_step45.npy"))
Y_test = np.load(os.path.join("/content","drive","My Drive","Colab Notebooks","lstm_data", "out_step45.npy"))
X_test = X_test[:,0:5]
Y_test = Y_test[:,0:5]

print(X_valid.shape, Y_valid.shape, X_test.shape, Y_test.shape)

# building model
model = Sequential()
model.add(Bidirectional(LSTM(40, return_sequences=True), input_shape=(None,23000)))
model.add(Bidirectional(LSTM(20, return_sequences=True)))
model.add(TimeDistributed(Dense(128, activation="relu")))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(Dense(61, activation="softmax")))
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              #optimizer=tf.keras.optimizers.Adadelta(lr=1.0),
              optimizer=tf.keras.optimizers.SGD(lr=0.0052),
              #optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()

# callbacks
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

checkpoint_path = "/content/drive/My Drive/Colab Notebooks/lstm_models/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_loss',mode='min', verbose=1, save_best_only=True)
csv_logger = CSVLogger('/content/drive/My Drive/Colab Notebooks/lstm_models/training1.log', append=True, separator=',')

# training
history = model.fit_generator(
    generate_array(mode="train"),
    steps_per_epoch = 863,
    validation_data = (X_valid, Y_valid),
    validation_steps = 863,
    epochs = 200,
    verbose=1,
    shuffle=True,
    initial_epoch = 0,
    callbacks=[es, cp_callback, csv_logger, PlotLossesKeras()])
	
# plot loss and accuracy graphs for visualisation
val_acc = history.history['val_acc']
acc = history.history['acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epoch_num = np.arange(0, len(val_acc), dtype=int)
plot1, = plt.plot(epoch_num, acc)
plot2, = plt.plot(epoch_num, val_acc)
plt.legend([plot1, plot2],['training accuracy', 'validation accuracy'])
plt.show()
plot1, = plt.plot(epoch_num, loss)
plot2, = plt.plot(epoch_num, val_loss)
plt.legend([plot1, plot2],['training loss', 'validation loss'])

# send message to telegram app on phone
telegram("LSTM bidirectional done!")

# evaluating model with test and validation set
train_result = model.evaluate_generator(
    generate_array(mode="train"), 
    steps = 863,
    verbose=0)
print('training loss:',train_result[0])
print('training accuracy:', train_result[1])

valid_result = model.evaluate(X_valid, Y_valid, verbose=0)
print('validation loss:',valid_result[0])
print('validation accuracy:', valid_result[1])

test_result = model.evaluate(X_test, Y_test, verbose=0)
print('testing loss:',test_result[0])
print('testing accuracy:', test_result[1])

message = "LSTM\n" + "test_loss :" + str(test_result[0]) + "\n" + "test_acc :" + str(test_result[1])  + "\n" + "valid_loss :" + str(valid_result[0]) + "\n" + "valid_acc :" + str(valid_result[1]) + "\n" + "train_loss :" + str(train_result[0]) + "\n" + "train_acc :" + str(train_result[1]) 
telegram(message)

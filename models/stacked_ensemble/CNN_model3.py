''' 
Author : Shalom Lee Li-Ling
Date: 10/10/2019
Title: models/stacked_ensemble/CNN_model3.py
Code Version: 
Availability: 
''' 
'''
This file is extracted from Google Colab notebook cnn_model3.ipynb (sllee41, neenyabigki7). 
It trains one of the stacked ensemble model (third model) as described in the final report
'''
import numpy as np 
import math
import tensorflow as tf
import random
import copy
import os
import requests
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
from livelossplot import PlotLossesKeras
import matplotlib.pyplot as plt
%matplotlib inline

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

# function to extract all data required
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
    print(x.shape, y.shape)
    return x, y

# define train, test, validation ratio and split data. First 30 scenes for level 0 training
# next 5 scenes for level 0 validation and next 5 scenes for level 0 testing
X, y = get_valid_data(mode="train")
X_train = X[:, 0:30, :, :, :]
y_train = y[:, 0:30, :]
X_valid = X[:, 30:35, :, :, :]
y_valid = y[:, 30:35, :]
X_test = X[:, 35:, :, :, :]
y_test = y[:, 35:,:]

# reshape for CNN input and output
X_train = X_train.reshape(-1, 46, 500, 1)
X_valid = X_valid.reshape(-1, 46, 500, 1)
X_test = X_test.reshape(-1, 46, 500, 1)
y_train = y_train.reshape(-1, 61)
y_valid = y_valid.reshape(-1, 61)
y_test = y_test.reshape(-1, 61)

nrows = X_train.shape[1]
ncolumns = X_train.shape[2]

input_shape = (nrows, ncolumns, 1)

print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'valid samples')
print("Finished reading datasets")

numClasses = y_train.shape[1]

# define cnn model
def create_model(numClasses):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Conv2D(100, kernel_size=(24,24),
              activation='relu',input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(50, (12, 12), padding="SAME", activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(6, 6)))
    model.add(tf.keras.layers.Conv2D(20, (10, 10), padding="SAME", activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4, activation='relu', kernel_regularizer = regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(numClasses, activation='softmax'))
  
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
              #optimizer=tf.keras.optimizers.Adadelta(lr=1.0),
              #optimizer=tf.keras.optimizers.SGD(lr=0.00052),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
    
    return model

model = create_model(numClasses)
model.summary()

# callbacks
es = EarlyStopping(monitor='val_loss', mode='min',verbose=1, patience=20)

checkpoint_path = "/content/drive/My Drive/Colab Notebooks/trial3_data/cnn_data_trial3data/cp3_short-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_loss',mode='min', verbose=1, save_best_only=False)

csv_logger = CSVLogger('/content/drive/My Drive/Colab Notebooks/trial3_data/cnn_data_trial3data/training3_short.log', append=True, separator=',')

batch_size = 16
epochs = 100

# training
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    initial_epoch = 0,
                    verbose=1,
                    shuffle=True,
                    validation_data=(X_valid, y_valid), callbacks=[es, cp_callback, csv_logger, PlotLossesKeras()])                    
   
# save model and its weights 
model.save("/content/drive/My Drive/Colab Notebooks/trial3_data/cnn_data_trial3data/model_3_3_short.h5")
print("Saved model_3.h5\n")

model.save_weights("/content/drive/My Drive/Colab Notebooks/trial3_data/cnn_data_trial3data/model3_3_shortweights.h5")
model_json = model.to_json()
with open("/content/drive/My Drive/Colab Notebooks/trial3_data/cnn_data_trial3data/model3_3_shortarchitecture.json", "w") as json_file:
    json_file.write(model_json)

# evaluate trained model and plot loss + accuracy graphs
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
valid_loss, valid_acc = model.evaluate(X_valid, y_valid, verbose=0)

val_acc = history.history['val_acc']
acc = history.history['acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
np.savez("/content/drive/My Drive/Colab Notebooks/trial3_data/cnn_data_trial3data/val_acc_3_short.npz",loss=loss, acc=acc, val_loss=val_loss, val_acc=val_acc)

epoch_num = np.arange(0, len(val_acc), dtype=int)

plot1, = plt.plot(epoch_num, acc)
plot2, = plt.plot(epoch_num, val_acc)
plt.legend([plot1, plot2],['training accuracy', 'validation accuracy'])
plt.show()

plot1, = plt.plot(epoch_num, loss)
plot2, = plt.plot(epoch_num, val_loss)
plt.legend([plot1, plot2],['training loss', 'validation loss'])
plt.show()

# send message via telegram app to indicate model has finished training
message = "cnn for reinforcement\n" + "test_loss :" + str(test_loss) + "\n" + "test_acc :" + str(test_acc)  + "\n" + "valid_loss :" + str(valid_loss) + "\n" + "valid_acc :" + str(valid_acc) + "\n" + "train_loss :" + str(train_loss) + "\n" + "train_acc :" + str(train_acc) 
telegram(message)

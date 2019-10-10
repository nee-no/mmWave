''' 
Author : Shalom Lee Li-Ling (modified Aldebaro codes as stated below)
Date: 10/10/2019
Title: models/baseline/baseline2.py 
Code Version: 
Availability: 
''' 
'''
CNN model with division of dataset not by scenes but randomly. More dropouts and regularization than baseline1.
Baseline2 in final report.
'''
from __future__ import print_function
from sklearn.preprocessing import minmax_scale
import math
import numpy as np
import tensorflow as tf
import numpy as np
import os
import csv
import re
import sys
import copy
#one can disable the imports below if not plotting / saving
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint

# loading data
def load_raw_data():
	ori_file = os.path.join("/home","englab5510","PycharmProjects","mmWave2","episodeData","allEpisode.npz")
	ori_data = np.load(ori_file)

	num_train_ep = 116
	
	pos_mat = ori_data['position_matrix_array']
	best_ray = ori_data['best_ray_array']

	pos_mat[pos_mat < 0 ] = -1
	pos_mat[pos_mat > 0 ] = 1

	t1 = ori_data['t1s_array']
	
	return pos_mat, best_ray, t1

# open validity file get all valid data
def validity_check(pos_mat):
	valid_invalid_dir = os.path.join("/home","englab5510","software","MIMO_5G_Data","5gm-data-2","Valid_and_Invalid_channels","list1_valids_and_invalids.csv")

	with open(valid_invalid_dir, 'r') as f:
		validity = list(csv.reader(f, delimiter=","))
	validity = np.array(validity[0:], dtype=np.str)
	
	# record valid and invalid cases in an array. To be used for comparison later
	array = np.zeros((pos_mat.shape[0], pos_mat.shape[1], pos_mat.shape[2]))

	for i in range(validity.shape[0]):
		episode = int(validity[i][1])
		scene = int(validity[i][2])
		receiver = int(validity[i][3])
		valid = validity[i][0]

	if re.match('V', valid, flags=0):
		array[episode][scene][receiver] = 1
	else:
		array[episode][scene][receiver] = 0
	
	# create a matrix to keep track of valid cases and cases where there's 50 scenes.
	validity_matrix = np.zeros((pos_mat.shape[0], pos_mat.shape[2], pos_mat.shape[1]))
	valid_count = np.zeros((pos_mat.shape[0], pos_mat.shape[2])) # to count valid scenes for each receiver

	# loop through array variable, count valid receivers
	for index, x in np.ndenumerate(array):
		if x == 1: # valid cases only
			valid_count[index[0], index[2]] += 1
			validity_matrix[index[0], index[2], index[1]] = 1
			
	# count number of times 50 scene appears for each (ep, rx) pair
	# unique, counts = np.unique(valid_count, return_counts=True)
	# dictionary = dict(zip(unique, counts))
	# print(dictionary)
	
	return valid_count, validity_matrix

# return valid data
def get_valid_data(pos_mat, best_ray, t1, valid_count, validity_matrix):
	position_matrix_all = []
	best_ray_all = []
	t1_all = []

	for i in range(0, pos_mat.shape[0]): # ep
		for k in range(0, pos_mat.shape[2]): # rx
			if valid_count[i, k] >= 1:
				temp = np.where(validity_matrix[i,k,:] != 0) # validity_matrix:(ep, rx, scene)
				
				best_ray_data = best_ray[i,temp,k]
				best_ray_data = best_ray_data.reshape(best_ray_data.shape[1], best_ray_data.shape[2])

			if np.any(np.argwhere(np.isnan(best_ray_data))):
				continue

			position_data = pos_mat[i,temp,k,:,:]
			position_data = position_data.reshape(position_data.shape[1], position_data.shape[2], position_data.shape[3])
			position_data = position_data.reshape(position_data.shape[0], position_data.shape[1], position_data.shape[2], 1)

			t1_data = t1[i,temp,k,:,:]
			t1_data = t1_data.reshape(t1_data.shape[1], t1_data.shape[2], t1_data.shape[3]) # scene,16, 16
			t1_data = t1_data.reshape(t1_data.shape[0], -1)

			while (best_ray_data.shape[0] < 50):
				position_data = np.insert(position_data, [0], position_data[0,:,:], axis=0)
				best_ray_data = np.insert(best_ray_data, [0], best_ray_data[0,:], axis=0)
				t1_data = np.insert(t1_data, [0], t1_data[0,:], axis=0)
			
			position_matrix_all.append(position_data)
			best_ray_all.append(best_ray_data)
			t1_all.append(t1_data)
			#print(position_data.shape, best_ray_data.shape, t1_data.shape)
	
	# convert all to np array, reshape into (num_valid_cases of (ep,rx), scene_num, 23000) for pos matrix
	position_matrix_all = np.array(position_matrix_all)
	best_ray_all = np.array(best_ray_all)
	t1_all = np.array(t1_all)

	# reshape all data
	numUPAAntennaElements = 4*4
	
	#convert output (i,j) to single number (the class label) and eliminate pairs that do not appear
	temp = best_ray_all.reshape((-1,2))
	full_y = (best_ray_all[:,:,0] * numUPAAntennaElements + best_ray_all[:,:,1]).astype(np.int)
	temp = (temp[:,0] * numUPAAntennaElements + temp[:,1]).astype(np.int)
	
	classes = set(temp)
	y_train = np.zeros([best_ray_all.shape[0], best_ray_all.shape[1]])
	
	t1_data_valid = np.zeros((best_ray_all.shape[0], best_ray_all.shape[1], len(classes)))
	
	for idx, cl in enumerate(classes): #map in single index, cl is the original class number, idx is its index
		t1_data_valid[:,:,idx] = t1_all[:,:,cl] # extract power of valid
		cl_idx = np.nonzero(full_y == cl)
		y_train[cl_idx[0], cl_idx[1]] = idx
	ratio = [40, 45, 50]
	
	y_dat = np.empty((y_train.shape[0], 50, len(classes)))
	
	for i in range(0, y_train.shape[0]):
		y_dat[i, :] = tf.keras.utils.to_categorical(y_train[i,:], len(classes))
	print(position_matrix_all.shape, y_dat.shape)
	
	return position_matrix_all, y_dat

# get valid data
pos_mat, best_ray, t1 = load_raw_data()
valid_count, validity_matrix = validity_check(pos_mat)
position_matrix_all, y_dat = get_valid_data(pos_mat, best_ray, t1, valid_count, validity_matrix)

# separate valid data into 80% train, 10% validation, 10% test
X = position_matrix_all.reshape((-1, 46, 500, 1))
Y = y_dat.reshape((-1, 61))
ratio = X.shape[0]*np.array([0.8, 0.1, 0.1])
ratio[1] = ratio[0]+ratio[1]
ratio[2] = ratio[1]+ratio[2]
print(X.shape, ratio)
X_train = X[0:int(ratio[0])]
X_valid = X[int(ratio[0]):int(ratio[1])]
X_test = X[int(ratio[1]):int(ratio[2])]
Y_train = Y[0:int(ratio[0])]
Y_valid = Y[int(ratio[0]):int(ratio[1])]
Y_test = Y[int(ratio[1]):int(ratio[2])]

''' 
Author : Aldebaro Klautau, Pedro Batista, Nuria Gonzalez-Prelcic, Yuyang Wang and Robert W. 
Date: 2018 
Title: 5gm-beam-selection/classification/deep_ann_classifier.py 
Code Version: 
Availability: https://github.com/lasseufpa/5gm-beam-selection/blob/master/classification/deep_ann_classifier.py 
Modified by : Shalom Lee
''' 
'''Trains a simple deep NN on ITA paper drop based dataset.
Adapted by AK: Feb 7, 2018 - I took out the graphics. Uses Pedro's datasets with 6 antenna elements per U
PA, which has 26 classes.
See for explanation about convnet and filters:
https://datascience.stackexchange.com/questions/16463/what-is-are-the-default-filters-used-by-keras-convo
lution2d
and
http://cs231n.github.io/convolutional-networks/
'''
batch_size = 32
epochs = 150

numUPAAntennaElements=4*4 #4 x 4 UPA

numClasses = Y_train.shape[1] #total number of labels

train_nexamples=X_train.shape[0]
test_nexamples=X_test.shape[0]
nrows=X_train.shape[1]
ncolumns=X_train.shape[2]

print('test_nexamples = ', test_nexamples)
print('train_nexamples = ', train_nexamples)
print('input matrices size = ', nrows, ' x ', ncolumns)
print('numClasses = ', numClasses)

#here, do not convert matrix into 1-d array
#X_train = X_train.reshape(train_nexamples,nrows*ncolumns)
#X_test = X_test.reshape(test_nexamples,nrows*ncolumns)

input_shape = (nrows, ncolumns, 1) #the input matrix with the extra dimension requested by Keras

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_valid.shape[0], 'validation samples')
print(X_test.shape[0]+X_train.shape[0]+X_valid.shape[0], 'total samples')
print("Finished reading datasets")

# declare model Convnet with two conv1D layers following by MaxPooling layer, and two dense layers
# Dropout layer consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.
# Creates a session with log_device_placement set to True.
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(100, kernel_size=(10,10), activation='relu', input_shape=input_shape))
model.add(tf.keras.layers.Conv2D(50, (12, 12), padding="SAME", activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(6, 6)))
model.add(tf.keras.layers.Conv2D(20, (10, 10), padding="SAME", activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(numClasses, activation='softmax'))

# callbacks
es = EarlyStopping(monitor='val_loss', mode='min',verbose=1, patience=50)
checkpoint_path = "baseline_models/cp2-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
csv_logger = CSVLogger('baseline_models/training2.log', append=True, separator=',')
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss',mode='min', verbose=1, save_best_only=False)

model.compile(loss=tf.keras.losses.categorical_crossentropy, 
	optimizer=tf.keras.optimizers.Adadelta(),
	metrics=['accuracy'])

history = model.fit(X_train, Y_train,
	batch_size=batch_size,
	epochs=epochs,
	verbose=1,
	shuffle=True,
	validation_data=(X_valid, Y_valid),
	callbacks=[es, cp_callback, csv_logger])

# print results
score = model.evaluate(X_test, Y_test, verbose=0)
print(model.metrics_names)
print(score)

model.save('baseline_models/baseline_deep_ann_model2.h5')
model.save_weights("baseline_models/baseline_deep_ann_model_weight2.h5", overwrite=True)

with open('baseline_models/baseline_deep_ann_model_architecture2.json', 'w') as f:
	f.write(model.to_json())
	
val_acc = history.history['val_acc']
acc = history.history['acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

np.savez('baseline_models/baseline_deep_ann_val_acc2.npz',validation_acc=val_acc, testing_acc=acc, validation_loss=val_loss, testing_loss=loss)

# enable if want to plot images
if True:
	# from tf.keras.utils import plot_model
	# install graphviz: sudo apt-get install graphviz and then pip install related packages
	plot_model(model, to_file='baseline_models/baseline_deep_ann_model2.png', show_shapes = True)
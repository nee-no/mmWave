''' 
Author : Shalom Lee Li-Ling
Date: 10/10/2019
Title: dataset/lstm_data.py
Code Version: 
Availability: 
''' 
'''
This python file extracts data for lstm training such that there is overlapping windows of scenes. 
Eg, one 'step0' saves data from scene 0-9, another 'step1' saves data from scene 1-10. 
'''

import math
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import numpy as np
import os
import csv
import re
import sys

# loading data
def load_raw_data():
    ori_file = os.path.join("/home","englab5510","PycharmProjects","mmWave2","episodeData","allEpisode.npz")
    ori_data = np.load(ori_file)

    num_train_ep = 116

    pos_mat = ori_data['position_matrix_array']
    best_ray = ori_data['best_ray_array']
    pos_mat[pos_mat < 0 ] = -1
    pos_mat[pos_mat > 0 ] = 1
    #path_gains = ori_data['path_gains_array']
    #departure_angle = ori_data['departure_angles_array']
    #arrival_angle = ori_data['arrival_angles_array']
    t1 = ori_data['t1s_array']

    #return pos_mat, best_ray, path_gains, departure_angle, arrival_angle, t1
    return pos_mat, best_ray, t1

# open validity file
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
        if x == 1:  # valid cases only
            valid_count[index[0], index[2]] += 1
            validity_matrix[index[0], index[2], index[1]] = 1

    # count number of times 50 scene appears for each (ep, rx) pair
    # unique, counts = np.unique(valid_count, return_counts=True)
    # dictionary = dict(zip(unique, counts))
    # print(dictionary)

    return valid_count, validity_matrix

def get_valid_data(pos_mat, best_ray, t1, valid_count, validity_matrix):
    in_folder = os.path.join("/home","englab5510","PycharmProjects","mmWave1","LSTM_notebooks", "lstm_data")

    # all
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
                
                while (best_ray_data.shape[0] < 50):
                    position_data = np.insert(position_data, [0], position_data[0,:,:], axis=0)
                    best_ray_data = np.insert(best_ray_data, [0], best_ray_data[0,:], axis=0)

                position_matrix_all.append(position_data)
                best_ray_all.append(best_ray_data)
                    
                
    # convert all to np array, reshape into (num_valid_cases of (ep,rx), scene_num, 23000) for pos matrix
    position_matrix_all = np.array(position_matrix_all)
    best_ray_all = np.array(best_ray_all)
    
    # reshape all data
    numUPAAntennaElements = 4*4
    
    #convert output (i,j) to single number (the class label) and eliminate pairs that do not appear
    temp = best_ray_all.reshape((-1,2))
    full_y = (best_ray_all[:,:,0] * numUPAAntennaElements + best_ray_all[:,:,1]).astype(np.int)
    temp = (temp[:,0] * numUPAAntennaElements + temp[:,1]).astype(np.int)

    classes = set(temp)
    y_train = np.zeros([best_ray_all.shape[0], best_ray_all.shape[1]])
    
    for idx, cl in enumerate(classes): #map in single index, cl is the original class number, idx is its index
        cl_idx = np.nonzero(full_y == cl)
        y_train[cl_idx[0], cl_idx[1]] = idx
    ratio = [40, 45, 50]
    
    y_dat = np.empty((y_train.shape[0], 50, len(classes)))
    for i in range(0, y_train.shape[0]):
        y_dat[i, :] = tf.keras.utils.to_categorical(y_train[i,:], len(classes))
    
    # saving for training data
    a = 0
    timesteps = 10
    max_train_scenes = 50
    max_a = max_train_scenes - timesteps
    position_matrix_all = position_matrix_all.reshape((position_matrix_all.shape[0], position_matrix_all.shape[1], -1))
                                                      
    for j in range(0, max_a+1):
        # save to file
        in_filename = "in_step" + str(j)
        position_data = position_matrix_all[:,a:a+timesteps,:]
        
        # save in shape (scene, channel, row, column)
        np.save(os.path.join(in_folder, in_filename), position_data)
        
        out_filename = "out_step" + str(j)
        y_data = y_dat[:,a:a+timesteps,:]
                                                      
        np.save(os.path.join(in_folder, out_filename), y_data)
        
        a += 1
                                                      
    for i in range(41, 50):
        position_data = position_matrix_all[:, i:, :]
        y_data = y_dat[:, i:, :]
        shapes_data = position_data.shape[1]
        while position_data.shape[1] < timesteps:
            position_data = np.insert(position_data, [shapes_data], position_data[position_data.shape[0]-1,position_data.shape[1]-1, :], axis=1)
            y_data =  np.insert(y_data, [shapes_data], y_dat[y_dat.shape[0]-1,y_dat.shape[1]-1,:], axis=1)
            
        in_filename = "in_step" + str(i)
        np.save(os.path.join(in_folder, in_filename), position_data)
        
        out_filename = "out_step" + str(i)
        np.save(os.path.join(in_folder, out_filename), y_data)
   
    return t1_data_valid, position_matrix_all, best_ray_all

pos_mat, best_ray, t1 = load_raw_data()
valid_count, validity_matrix = validity_check(pos_mat)
t1_data, position_data, best_t1 = get_valid_data(pos_mat, best_ray, t1, valid_count, validity_matrix)
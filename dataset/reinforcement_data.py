''' 
Author : Shalom Lee Li-Ling
Date: 10/10/2019
Title: dataset/reinforcement_data.py
Code Version: 
Availability: 
''' 
'''
This python file extracts data for reinforcement training such that the received power, position and best beam pair of each episode (893 in total) is saved in different folders. This is done for easier retrival during training and preventing overloading of memory.
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
    in_train_folder = os.path.join("/home","englab5510","PycharmProjects","mmWave1","reinforcement_data", "trial3","input_train")
    out_train_folder = os.path.join("/home","englab5510","PycharmProjects","mmWave1", "reinforcement_data", "trial3","output_train")

    in_test_folder = os.path.join("/home","englab5510","PycharmProjects","mmWave1","reinforcement_data", "trial3","input_test")
    out_test_folder = os.path.join("/home","englab5510","PycharmProjects","mmWave1", "reinforcement_data", "trial3","output_test")

    in_valid_folder = os.path.join("/home","englab5510","PycharmProjects","mmWave1","reinforcement_data", "trial3","input_valid")
    out_valid_folder = os.path.join("/home","englab5510","PycharmProjects","mmWave1", "reinforcement_data", "trial3","output_valid")
    
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
                t1_data = t1[i,temp,k,:,:]
                t1_data = t1_data.reshape(t1_data.shape[1], t1_data.shape[2], t1_data.shape[3]) # scene, 16, 16
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
    
    for j in range(0, position_matrix_all.shape[0]):
        in_filename = "sample" + str(j)
        position_data = position_matrix_all[j,0:50,:,:]
        np.save(os.path.join(in_train_folder, in_filename), position_data[0:ratio[0]])
        np.save(os.path.join(in_valid_folder, in_filename), position_data[ratio[0]:ratio[1]])
        np.save(os.path.join(in_test_folder, in_filename), position_data[ratio[1]:ratio[2]])
        
        out_filename = "category" + str(j)
        y_data = tf.keras.utils.to_categorical(y_train[j,:], len(classes))
        np.save(os.path.join(out_train_folder, out_filename), y_data[0:ratio[0]])
        np.save(os.path.join(out_valid_folder, out_filename), y_data[ratio[0]:ratio[1]])
        np.save(os.path.join(out_test_folder, out_filename), y_data[ratio[1]:ratio[2]])
        
        out_filename = "t1_" + str(j)
        t1_data = t1_data_valid[j,:,:]
        np.save(os.path.join(out_train_folder, out_filename), t1_data[0:ratio[0]])
        np.save(os.path.join(out_valid_folder, out_filename), t1_data[ratio[0]:ratio[1]])
        np.save(os.path.join(out_test_folder, out_filename), t1_data[ratio[1]:ratio[2]])
    
    print(y_train.shape, t1_data_valid.shape, position_matrix_all.shape)
    
    return t1_data_valid, position_matrix_all, best_ray_all


pos_mat, best_ray, t1 = load_raw_data()
valid_count, validity_matrix = validity_check(pos_mat)
t1_data, position_data, best_t1 = get_valid_data(pos_mat, best_ray, t1, valid_count, validity_matrix)
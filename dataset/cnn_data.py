''' 
Author : Shalom Lee Li-Ling
Date: 10/10/2019
Title: dataset/reinforcement_data.py
Code Version: 
Availability: 
''' 
'''
This python file extracts data for cnn supervised training from the extracted reinforcement data (reinforcecment_data.py).
It extracts the data and reshapes it such that the input and output data are ready for cnn training. 
'''
import numpy as np 
import math
import tensorflow as tf
import random
import copy
import os

def get_valid_data(mode): 
    in_train_folder = os.path.join("/home","englab5510","PycharmProjects","mmWave1","reinforcement_data", "trial3","input_train")
    out_train_folder = os.path.join("/home","englab5510","PycharmProjects","mmWave1", "reinforcement_data", "trial3","output_train")

    in_test_folder = os.path.join("/home","englab5510","PycharmProjects","mmWave1","reinforcement_data", "trial3","input_test")
    out_test_folder = os.path.join("/home","englab5510","PycharmProjects","mmWave1", "reinforcement_data", "trial3","output_test")

    in_valid_folder = os.path.join("/home","englab5510","PycharmProjects","mmWave1","reinforcement_data", "trial3","input_valid")
    out_valid_folder = os.path.join("/home","englab5510","PycharmProjects","mmWave1", "reinforcement_data", "trial3","output_valid")
    
    if (mode=="train"):
        x = np.zeros((863, 40, 46, 500, 1))
        y = np.zeros((863, 40, 61))
        t1 = np.zeros((863, 40, 61))
        
    elif (mode=="test"):
        x = np.zeros((863, 5, 46, 500, 1))
        y = np.zeros((863, 5, 61))
        t1 = np.zeros((863, 5, 61))
        
    else:
        x = np.zeros((863, 5, 46, 500, 1))
        y = np.zeros((863, 5, 61))
        t1 = np.zeros((863, 40, 61))
        
        
    for j in range(0, 863):
        in_filename = "sample" + str(j) + ".npy"
        out_filename = "category" + str(j) + ".npy"
        
        if (mode=="train"):
            data_x = np.load(os.path.join(in_train_folder, in_filename))
            data_y = np.load(os.path.join(out_train_folder, out_filename))
            
        elif (mode=="test"):
            data_x = np.load(os.path.join(in_test_folder, in_filename))
            data_y = np.load(os.path.join(out_test_folder, out_filename))
            
        else:
            data_x = np.load(os.path.join(in_valid_folder, in_filename))
            data_y = np.load(os.path.join(out_valid_folder, out_filename))
            
        x[j, :, :] = data_x
        y[j, :, :] = data_y
    
    return x, y

# define train, test, validation ratio and split data
X_train, y_train = get_valid_data(mode="train")
X_test, y_test = get_valid_data(mode="test")
X_valid, y_valid = get_valid_data(mode="valid")
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_valid.shape, y_valid.shape)

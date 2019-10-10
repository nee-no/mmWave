#https://github.com/openai/gym-soccer/blob/master/gym_soccer/envs/soccer_env.py
from livelossplot.keras import PlotLossesCallback
from matplotlib import pyplot as plt

import numpy as np
import tensorflow as tf
import random
import csv
import math
import os
import copy
from keras.utils import plot_model
import keras
import string
import pickle
import re
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


ENV_NAME = "mmWave_v0"

env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

input_shape = (46, 500, 1)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(100, kernel_size=(10,10),activation='relu', input_shape=input_shape))
model.add(tf.keras.layers.Conv2D(50, (12, 12), padding="SAME", activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(6, 6)))
model.add(tf.keras.layers.Conv2D(20, (10, 10), padding="SAME", activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(62, activation='softmax'))
model.summary()

model.compile(loss=tf.keras.losses.categorical_crossentropy,
    #optimizer=tf.keras.optimizers.Adadelta(lr=1.0),
    optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0),
    #optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy'])

memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)

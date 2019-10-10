import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import logging
import random
import math
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# https://github.com/openai/gym-soccer/blob/master/gym_soccer/envs/soccer_env.py

class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._num_ep = 862
        self.count = None # to keep count of the number of states
        self._states_available = None # keeps record of all possible states of current episode
        self._power_available = None # keeps record of all possible power levels of current episode
        self._actions_available = None # keep record of best action
        self._tx_rx_pair = None # keeps record of which action number correspond to which tx, rx pair
        self.action = None
        self.state = None
        self.reward = None
        self.complete = None
        self.tx = None
        self.rx = None
        self._ep_counter = 0
        self._acc_counter = 0

        self._seed()
        
        self.action_space = spaces.Discrete(61) # 62 actions since only 62 valid beampairs 
        self.observation_space = spaces.Box(low=-1, high=1, shape=(46,500,1), dtype=np.int8) # 46,500
      
    def _seed(self, seed_num=0): # this is so that you can get consistent results
        pass                     # optionally, you could add: random.seed(random_num)
        return
        
    def step(self, action):
        self._update_state(action)
        return self.state, self.reward, self.complete, {}
        
    def reset(self):
        ep_ind = int(self._ep_counter)
        self._acc_counter = 0
        if ep_ind < self._num_ep:
            self._ep_counter += 1 # testing
        else:
            self._ep_counter = 0
        
        #ep_ind = random.randint(0, int(self._num_ep)) # training

        # load position data and t1 for selected episode
        self._states_available = np.load( '/home/englab5510/PycharmProjects/mmWave1/reinforcement_data/trial3/input_test/sample{}.npy'.format(ep_ind))
        self._power_available = np.load('/home/englab5510/PycharmProjects/mmWave1/reinforcement_data/trial3/output_test/t1_{}.npy'.format(ep_ind))
        # power saved in (scene, 62 powers)
        self._actions_available = np.load( '/home/englab5510/PycharmProjects/mmWave1/reinforcement_data/trial3/output_test/category{}.npy'.format(ep_ind))
        # actions saved in (scene, onehot 62 action)
        
        #print(self._states_available.shape)
        #self._states_available = self._scale(self._states_available, 0, 1) # normalising
        
        self.count = 0
        
        ideal_action = self._actions_available[self.count]
        ideal_action = np.where(ideal_action==1)        
        
        power_state = self._power_available[self.count, :]
        a = power_state[power_state==0]
        
        while a.size != 0 | self._is_empty(ideal_action[0]):
            self._ep_counter += 1
            ep_ind = int(self._ep_counter)
            self._states_available = np.load('/home/englab5510/PycharmProjects/mmWave1/reinforcement_data/trial3/input_test/sample{}.npy'.format(ep_ind))
            self._power_available = np.load( '/home/englab5510/PycharmProjects/mmWave1/reinforcement_data/trial3/output_test/t1_{}.npy'.format(ep_ind))
            self._actions_available = np.load( '/home/englab5510/PycharmProjects/mmWave1/reinforcement_data/trial3/output_test/category{}.npy'.format(ep_ind))
            power_state = self._power_available[self.count, :]
            
            a = power_state[power_state==0]
            ideal_action = self._actions_available[self.count]
            ideal_action = np.where(ideal_action==1)
    
        print(ep_ind, ideal_action, ideal_action==[0])
            
        # check if all scenes is completed
        if self._power_available.shape[0]==1:
            self.complete = 1
        else:
            self.complete = 0
        
        #self._states_available = self._scale(self._states_available, 0, 1) # scale between 0 and 1
        self.state = self._states_available[0] # return first position matrix
        self.reward = None # reset reward cos no action selected
        #self._draw_canvas()
        return self.state
    
    def _is_empty(self, structure):
        if structure:
            return False
        else:
            return True
    
    def _scale(self, X, x_min, x_max):
        # https://datascience.stackexchange.com/questions/39142/normalize-matrix-in-python-numpy
        nom = (X-X.min(axis=0))*(x_max-x_min)
        denom = X.max(axis=0) - X.min(axis=0)
        denom[denom==0] = 1
        return x_min + nom/denom  
        
    def render(self, mode='human',close=False):
        state = self.state
        state = state.reshape((state.shape[0], -1))
        ax = plt.subplot(121)
        ax.imshow(state.T, origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.draw()
        plt.pause(0.1)
        
        
    def _update_state(self, action):
        # updates the current state to next state
        old_state = self.state
        new_state = self._states_available[self.count, :, :]
        ideal_action = self._actions_available[self.count]
        ideal_action = np.where(ideal_action==1)
        ideal_action = ideal_action[0][0]
        
        # check if episode had reached the end
        if (self._states_available.shape[0]-1)==self.count:
            complete = 1
        else:
            complete = 0

        if action==ideal_action:
            reward = 1
            self._acc_counter += 1
        else:
            reward = 0
        
        if complete:
            print('accuracy: ',self._acc_counter) # value is close to 5 if continuously choosing correct action
        
        # normalise according to paper
        power_state = self._power_available[self.count, :]

        temp = 20*np.log(power_state)
        temp[np.isinf(temp)] = 0
        
        y_s_r_i = power_state[int(action)]

        self.count += 1

        self.state = new_state
        self.complete = complete
        self.reward = reward
        self.canvas = self.state
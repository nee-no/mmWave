''' 
Author : Shalom Lee Li-Ling
Date: 10/10/2019
Title: models/reinforcement/method1_reward.py (original3_foo.py)
Code Version: 
Availability: 
''' 
'''
This file is what is explained in the final report. It contains all four classes used in reinforcement learning.
'''
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


# get new (valid and with 50 scenes) data
def split_data(num_poss_ep, valid_data, mode):
    if (mode == "test"):
        x_out = np.load('dataset4/reinforcement/x_test.npy')
        y_out = np.load('dataset4/reinforcement/t1_test.npy')
    else:
        x_out = np.load('dataset4/reinforcement/x_valid.npy')
        y_out = np.load('dataset4/reinforcement/t1_valid.npy')

    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], 1)
    y_out = y_out.reshape(y_out.shape[0], y_out.shape[1] * y_out.shape[2])
    return x_out, y_out


class Model:
    def __init__(self, num_states, num_actions, bat_size):
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = bat_size
        self.model = None
        self.acc = []
        self.loss = []

        # setup model
        self._define_model()

    def _define_model(self):
        input_shape = (num_states[0], num_states[1], 1)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(100, kernel_size=(10, 10), activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.Conv2D(50, (12, 12), padding="SAME", activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(6, 6)))
        model.add(tf.keras.layers.Conv2D(20, (10, 10), padding="SAME", activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(4, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.num_actions, activation='softmax'))
        model.summary()

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      # optimizer=tf.keras.optimizers.Adadelta(lr=1.0),
                      optimizer=tf.keras.optimizers.SGD(lr=0.052),
                      metrics=['accuracy'])
        self.model = model
        self._X_valid, self._Y_valid = split_data(num_poss_ep, 913, mode="valid")
        self._X_test, self._Y_test = split_data(num_poss_ep, 913, mode="test")

    def predict_batch(self, states):
        print("predict batch ", states.shape)
        states = np.reshape(states, (states.shape[0], 46, 500, 1))
        prediction = self.model.predict(states)
        return prediction

    def predict_action(self, states):
        states = np.reshape(states, (1, 46, 500, 1))
        prediction = self.model.predict(states)
        return prediction

    def fit_model(self, x, y, epoch_num, num_poss_ep):
        x = np.reshape(x, (x.shape[0], 46, 500, 1))
        history = self.model.fit(
            x,
            y,
            batch_size=self.batch_size,
            epochs=epoch_num,
            verbose=1,
            shuffle=True)
        # ,
        # validation_data=(self._X_valid, self._Y_valid))
        self.acc.append(history.history['acc'])
        self.loss.append(history.history['loss'])
        return history

    def evaluate_model(self, x, y):
        x = np.reshape(x, (x.shape[0], 46, 500, 1))
        _, train_acc = self.model.evaluate(x, y, verbose=0)
        _, test_acc = self.model.evaluate(self._X_test, self._Y_test, verbose=0)
        return train_acc, test_acc


class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)


class GameRunner:
    # model dynamics, agent action and training
    def __init__(self, model, env, memory, max_eps, min_eps, eps_decay, gamma, num_poss_ep, render=False, ):
        # self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._render = render
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = eps_decay
        self._eps = max_eps
        self._steps = 0
        self._gamma = gamma
        self.reward_store = []
        self._max_x_store = []

    def run(self):
        state = self._env.reset()
        tot_reward = 0
        while True:
            if self._render:
                plt.ion()
                plt.draw()
                self._env.render()

            action = self._choose_action(state)
            next_state, reward, done, count_state = self._env.step(action)

            if int(done):
                next_state = None
            self._memory.add_sample((state, action, reward, next_state))
            train_acc, test_acc = self._replay()

            # exponentially decay the epsilon value
            self._steps += 1
            # self._eps = self._min_eps + (self._max_eps - self._min_eps)*math.exp(-self._decay*self._steps)
            self._eps = self._min_eps + (self._max_eps - self._min_eps) * math.pow((1 - self._decay), self._steps)
            # self._eps *= self._decay

            # move agent to next state
            state = next_state
            tot_reward += reward

            if int(done):
                self.reward_store.append(tot_reward / count_state)
                break

        print("Step {}, Total reward: {}, Eps: {}, count_state: {}, train_acc: {}, test_acc: {}".format(self._steps,
                                                                                                        tot_reward,
     
    def _choose_action(self, state):
        if random.random() < self._eps:
            return np.random.randint(0, 256, dtype=int)
        else:
            return np.argmax(model.predict_action(state))

    def _replay(self):
        batch = self._memory.sample(self._model.batch_size)
        states = np.array([val[0] for val in batch])

        next_states = np.array([(np.zeros((1, self._model.num_states[0], self._model.num_states[1]))
                                 if val[3] is None else val[3]) for val in batch])

        print("_replay: ", next_states.shape)

        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below

        q_s_a_d = self._model.predict_batch(next_states)
        # setup training arrays

        x = np.zeros((len(batch), self._model.num_states[0], self._model.num_states[1]))
        y = np.zeros((len(batch), self._model.num_actions))

        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])

            x[i] = state
            y[i] = current_q

        epochs = 1
        history_val = self._model.fit_model(x, y, epochs, num_poss_ep)
        train_acc, test_acc = self._model.evaluate_model(x, y)
        return train_acc, test_acc


class Environment:
    _dictionary: None

    def __init__(self, num_poss_ep):
        self._num_ep = num_poss_ep
        self.count = None  # to keep count of the number of states
        self._states_available = None  # keeps record of all possible states of current episode
        self._power_available = None  # keeps record of all possible power levels of current episode
        self._tx_rx_pair = None  # keeps record of which action number correspond to which tx, rx pair
        self.action = None
        self.state = None
        self.reward = None
        self.complete = None
        self.tx = None
        self.rx = None

        filepath = os.path.join("/home", "englab5510", "software", "MIMO_5G_Data", "5gm-data", "position_mat_npz2","tx_rx.csv")
        self._read_action_pair(filepath)

    def _read_action_pair(self, file):
        f = open(file, 'r')
        reader = csv.reader(f)
        tx_rx_pair = {}
        for row in reader:
            tx_rx_pair[row[2]] = {'tx': row[0], 'rx': row[1]}

        self._tx_rx_pair = tx_rx_pair

    def _update_state(self, action):
        # updates the current state to next state
        new_state = self._states_available[self.count, :, :]

        # check if episode had reached the end
        if (self._states_available.shape[0] - 1) == self.count:
            complete = 1
            print(self.count, self._states_available.shape[0] - 1, complete)
        else:
            complete = 0
            print(self.count, self._states_available.shape[0] - 1, complete)

        # normalise according to paper
        self.tx = self._tx_rx_pair[str(int(action))]['tx']
        self.rx = self._tx_rx_pair[str(int(action))]['rx']
        power_state = self._power_available[self.count, :, :]

        temp = 20 * np.log(power_state)
        y_s_r_i = power_state[int(self.tx)][int(self.rx)]

        if y_s_r_i == 0:
            z_s_r_i = 0
        else:
            z_s_r_i = 20 * math.log(y_s_r_i)

        # remove negative infinity
        temp[temp < -10e6] = 0

        z_min = np.min(temp)
        z_max = np.max(temp)
        if (z_max - z_min != 0):
            z_s_r_i_bar = (z_s_r_i - z_min) / (z_max - z_min)
        else:
            z_s_r_i_bar = 0
        reward = z_s_r_i_bar

        self.count += 1

        self.state = new_state
        self.complete = complete
        self.reward = reward
        self.canvas = self.state

    def reset(self):
        # start new episode (randomly selects an episode from possible list of episodes) - only 80% used for training
        ep_ind = random.randint(0, int(self._num_ep))

        # load position data and t1 for selected episode
        self._states_available = np.load(
            'dataset4/input_train/sample{}.npy'.format(ep_ind))
        self._power_available = np.load(
            'dataset4/output_train/t1_{}.npy'.format(ep_ind))
        self.count = 0

        # check if all scenes is completed
        if self._power_available.shape[0] == 1:
            self.complete = 1
        else:
            self.complete = 0

        self.state = self._states_available[0]  # return first position matrix

        # self._draw_canvas()
        return self.state

    def step(self, action):
        self._update_state(action)
        # print(self._state_num, self.tx, self.rx)
        # self._draw_canvas()

        return self.state, self.reward, self.complete, self.count

    def render(self):
        state = self.state
        ax = plt.subplot(121)
        ax.imshow(state.T, origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.draw()
        plt.pause(0.1)


if __name__ == "__main__":

    num_poss_ep = 913
    env = Environment(num_poss_ep)
    num_states = [46, 500]
    num_actions = 16 * 16
    epoch = 1000
    max_memory = 23000
    batch_size = 128
    max_epsilon = 1.0
    min_epsilon = 0.1
    eps_decay = 0.00005
    gamma = 0.95

    model = Model(num_states, num_actions, batch_size)
    mem = Memory(max_memory)

    accuracy_list_ep = []
    loss_list_ep = []

    gr = GameRunner(model, env, mem, max_epsilon, min_epsilon, eps_decay, gamma, num_poss_ep)
    num_episodes = 100  # 23000
    cnt = 0

    while cnt < num_episodes:
        if cnt % 10 == 0:
            print('Episode {} of {}'.format(cnt + 1, num_episodes))
        gr.run()
        accuracy_list_ep.append(model.acc[-1])
        loss_list_ep.append(model.loss[-1])
        cnt += 1

        # include the epoch in the file name. (uses `str.format`)
        #checkpoint_path = "/content/drive/My Drive/Colab Notebooks/training_1/cp-{cnt:04d}.ckpt"
        #model.save(checkpoint_path)

    plt.plot(gr.reward_store)
    plt.show()


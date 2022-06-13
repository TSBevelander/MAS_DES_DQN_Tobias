import datetime
import random
import os

import keras
import pandas as pd

### DQN PART ###
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling2D, Activation, Flatten
from keras.optimizer_v2.adam import Adam
from collections import deque
import tensorflow as tf
from tqdm import tqdm
import time

EPISODES = 5_000
REPLAY_MEMORY_SIZE = 20_000
MODEL_NAME = '1stDQN'
epsilon = 1
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
ep_rewards0 = []
ep_rewards1 = []
ep_rewards2 = []
AGGREGATE_STATS_EVERY = 50
MIN_EPSILON = 0.001
EPSILON_DECAY = 0.999
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
MIN_REWARD = -1e10



import numpy as np

from IND_RL_PAS_Dynamic_Events import jobShop, Info, get_objectives

job_shop = jobShop()

import torch

actions = []
tardiness = []
Results = []
Results2 = []

DTI = []
DTFO = []
DTTO = []


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     self.step = 1
    #     self.writer = tf.summary.create_file_writer(log_dir)
    #
    # # Overriding this method to stop creating default log writer
    # def set_model(self, model):
    #     pass
    #
    # # Overrided, saves logs with our step number
    # # (otherwise every .fit() will start writing from 0th step)
    # def on_epoch_end(self, epoch, logs=None):
    #     self.update_stats(**logs)
    #
    # # Overrided
    # # We train for one batch only, no need to save anything at epoch end
    # def on_batch_end(self, batch, logs=None):
    #     pass
    #
    # # Overrided, so won't close writer
    # def on_train_end(self, _):
    #     pass
    #
    # # Custom method for saving own metrics
    # # Creates writer, writes custom metrics and closes writer
    # def update_stats(self, **stats):
    #     self._write_logs(stats, self.step)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.writer.flush()

# Agent class
class DQNAgent:
    def __init__(self, machine):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir = "logs/{}-{}-{}".format(MODEL_NAME, int(time.time()), machine))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Dense(18, input_dim=18, activation='linear'))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Dense(18, activation='linear'))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Dense(job_shop.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

agent1 = DQNAgent(1)

agent2 = DQNAgent(2)

agent0 = DQNAgent(0)


from torch.utils.tensorboard import SummaryWriter  # Tensorboard
def mean_calculator(list1,list2,value):
    sum = 0
    i = 0
    for ii in range(len(list1)):
        if list2[ii] > value:
            sum = sum + list1[ii]
            i += 1
    list_mean = sum/i
    return list_mean

def RandomAgent(observation):
    action = random.randint(0,6)
    return action

def AllAgent(observation, next_machine):
    action = random.randint(0,2)
    return action, next_machine

def set_manual_seed(manual_seed):
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

def observation_translate(observation, next_machine):
    if next_machine == 0:
        observation = observation[0:18]
    elif next_machine == 1:
        observation = observation[18:36]
    elif next_machine == 2:
        observation = observation[36:54]
    observation = np.array(observation)
    return observation

NUM_EPISODES = 1





setting = 0

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    observations1 = []
    observations2 = []
    observations0 = []
    actions0 = []
    actions1 = []
    actions2 = []
    rewards0 = []
    rewards1 = []
    rewards2 = []
    min_job_reached = False
    learning_done = [False,False,False]
    terminal = [False,False,False]

    # Update tensorboard step every episode
    agent0.tensorboard.step = episode
    agent1.tensorboard.step = episode
    agent2.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = [0,0,0]
    step = [1,1,1]

    # Reset environment and get initial state
    set_manual_seed(episode)
    job_shop = jobShop()
    done = False
    inf = Info(setting, True, 0)
    observation, next_machine = job_shop.reset(inf)

    # Reset flag and start iterating until episode ends
    while not done:
        observation = observation_translate(observation,next_machine)
        if min_job_reached == True:
            if next_machine == 0:
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(agent0.get_qs(observation))
                else:
                    # Get random action
                    action = np.random.randint(0, job_shop.ACTION_SPACE_SIZE)
            elif next_machine == 1:
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(agent1.get_qs(observation))
                else:
                    # Get random action
                    action = np.random.randint(0, job_shop.ACTION_SPACE_SIZE)
            elif next_machine == 2:
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(agent2.get_qs(observation))
                else:
                    # Get random action
                    action = np.random.randint(0, job_shop.ACTION_SPACE_SIZE)
        else:
            action = 0
        if next_machine == 0:
            observations0.append(observation)
            actions0.append(action)
        elif next_machine == 1:
            observations1.append(observation)
            actions1.append(action)
        elif next_machine == 2:
            observations2.append(observation)
            actions2.append(action)

        observation, terminal, reward, next_machine, prev_machine = job_shop.step(action, next_machine)

        if prev_machine == 0:
            rewards0.append(reward)
        elif prev_machine == 1:
            rewards1.append(reward)
        elif prev_machine == 2:
            rewards2.append(reward)

        if min_job_reached == True:
            episode_reward[prev_machine] += reward
            if (prev_machine == 0) and (learning_done[prev_machine] == True):
                agent0.update_replay_memory((observations0[-2], actions0[-1], observations0[-1], rewards0[-1]))
                agent0.train(terminal[prev_machine],step[prev_machine])
            elif (prev_machine == 1) and (learning_done[prev_machine] == True):
                agent1.update_replay_memory((observations1[-2], actions1[-1], observations1[-1], rewards1[-1]))
                agent1.train(terminal[prev_machine],step[prev_machine])
            elif (prev_machine == 2) and (learning_done[prev_machine] == True):
                agent2.update_replay_memory((observations2[-2], actions2[-1], observations2[-1], rewards2[-1]))
                agent2.train(terminal[prev_machine],step[prev_machine])
            step[prev_machine] += 1
            if terminal[prev_machine] == True:
                learning_done[prev_machine] = True

        if len(job_shop.tardiness) > job_shop.min_job:
            min_job_reached = True
        if sum(terminal) == 3:
            done = True

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards0.append(episode_reward[0])
    ep_rewards1.append(episode_reward[1])
    ep_rewards2.append(episode_reward[2])
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward0 = sum(ep_rewards0[-AGGREGATE_STATS_EVERY:])/len(ep_rewards0[-AGGREGATE_STATS_EVERY:])
        average_reward1 = sum(ep_rewards1[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards1[-AGGREGATE_STATS_EVERY:])
        average_reward2 = sum(ep_rewards2[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards2[-AGGREGATE_STATS_EVERY:])
        min_reward0 = min(ep_rewards0[-AGGREGATE_STATS_EVERY:])
        min_reward1 = min(ep_rewards1[-AGGREGATE_STATS_EVERY:])
        min_reward2 = min(ep_rewards2[-AGGREGATE_STATS_EVERY:])
        max_reward0 = max(ep_rewards0[-AGGREGATE_STATS_EVERY:])
        max_reward1 = max(ep_rewards1[-AGGREGATE_STATS_EVERY:])
        max_reward2 = max(ep_rewards2[-AGGREGATE_STATS_EVERY:])
        agent0.tensorboard.update_stats(reward_avg=average_reward0, reward_min=min_reward0, reward_max=max_reward0, epsilon=epsilon)
        agent1.tensorboard.update_stats(reward_avg=average_reward1, reward_min=min_reward1, reward_max=max_reward1, epsilon=epsilon)
        agent2.tensorboard.update_stats(reward_avg=average_reward2, reward_min=min_reward2, reward_max=max_reward2, epsilon=epsilon)
        ms, ft, mt, mxt, t1, t2, t3, mw, et = get_objectives(job_shop,inf.min_job,inf.max_job,job_shop.early_termination)
        Results.append([episode, average_reward0, average_reward1, average_reward2, min_reward0, min_reward1, min_reward2, max_reward0, max_reward1, max_reward2, ms, ft, mt, mxt, t1, t2, t3, mw, et])
        print(average_reward0, ft, mt)



        # Save model, but only when min reward is greater or equal a set value
        if min_reward0 >= MIN_REWARD:
            agent0.model.save(f'models/{MODEL_NAME}__{max_reward0:_>7.2f}max_{average_reward0:_>7.2f}avg_{min_reward0:_>7.2f}min__{int(time.time())}__machine0.model')
            MIN_REWARD = min_reward0
        if min_reward1 >= MIN_REWARD:
            agent1.model.save(f'models/{MODEL_NAME}__{max_reward1:_>7.2f}max_{average_reward1:_>7.2f}avg_{min_reward1:_>7.2f}min__{int(time.time())}__machine1.model')
            MIN_REWARD = min_reward1
        if min_reward2 >= MIN_REWARD:
            agent2.model.save(f'models/{MODEL_NAME}__{max_reward2:_>7.2f}max_{average_reward2:_>7.2f}avg_{min_reward2:_>7.2f}min__{int(time.time())}__machine2.model')
            MIN_REWARD = min_reward2

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)








# for setting in range(1):
#     for episode in range(NUM_EPISODES):
#         set_manual_seed(episode)
#         done = False
#         inf = Info(setting,True,0)
#         observation, next_machine = job_shop.reset(inf)
#         while not done:
#             action, next_machine = AllAgent(observation, next_machine)
#             if next_machine == 0:
#                 observations0.append(observation)
#                 actions0.append(action)
#             elif next_machine == 1:
#                 observations1.append(observation)
#                 actions1.append(action)
#             elif next_machine == 2:
#                 observations2.append(observation)
#                 actions2.append(action)
#
#             observation, done, reward, next_machine, prev_machine = job_shop.step(action, next_machine)
#             if prev_machine == 0:
#                 rewards0.append(reward)
#             elif prev_machine == 1:
#                 rewards1.append(reward)
#             elif prev_machine == 2:
#                 rewards2.append(reward)
#             if job_shop.env.now > 40:
#                 if prev_machine == 0:
#                     print(prev_machine,observations0[-2][58],observations0[-1][58],6+actions0[-2]*4,rewards0[-1])
#         ms, ft, mt, mxt, t1, t2, t3, mw, et = get_objectives(job_shop,inf.min_job,inf.max_job,job_shop.early_termination)
#         print("mean tardiness: " + str(mt) + ", mean flowtime: " + str(ft))
#         Results.append([setting, episode, ms, ft, mt, mxt, t1, t2, t3, mw, et])
#         print("end of setting: "+str(setting)+", episode: "+str(episode))


# rew1 = pd.DataFrame(data=rewards1)
# rew1.to_excel('rewards1.xlsx')
# rew2 = pd.DataFrame(data=rewards2)
# rew2.to_excel('rewards2.xlsx')
# rew3 = pd.DataFrame(data=rewards3)
# rew3.to_excel('rewards3.xlsx')
# arr = pd.DataFrame(data=job_shop.arrivaltimes)
# arr.to_excel('arrival.xlsx')
# due = pd.DataFrame(data=job_shop.dueDate)
# due.to_excel('duedate.xlsx')
# fin = pd.DataFrame(data=job_shop.jobfinishtime)
# fin.to_excel('finish.xlsx')
# prio = pd.DataFrame(data=job_shop.priority)
# prio.to_excel('priority.xlsx')

Res = pd.DataFrame(data=Results)
Res.to_excel('Results.xlsx')











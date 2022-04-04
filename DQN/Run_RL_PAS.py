import datetime
import random
import os
import pandas as pd

from make_environment import make_env
from agents import RandomAgent
import numpy as np
from torch.cuda import is_available
from RL_PAS_Dynamic_Events import jobShop, Info
from statistics import mean


import torch

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

def AllAgent(observation):
    action = 6
    return action

def RepairAgent(observation,MTTF,ratio):
    tim0 = observation[44]
    bro0 = observation[45]
    tim1 = observation[49]
    bro1 = observation[50]
    tim2 = observation[54]
    bro2 = observation[55]
    if bro0 == 1:
        bro0 = True
    else:
        bro0 = False
    if bro1 == 1:
        bro1 = True
    else:
        bro1 = False
    if bro2 == 1:
        bro2 = True
    else:
        bro2 = False
    ct = MTTF * ratio
    if ((tim0 <= ct)and(not bro0)) and ((tim1 <= ct)and(not bro1)) and ((tim2 <= ct)and(not bro2)):
        action = 6
    elif ((tim0 <= ct)and(not bro0)) and ((tim1 <= ct)and(not bro1)) and ((tim2 > ct) or (bro2)):
        action = 3
    elif ((tim0 <= ct)and(not bro0)) and ((tim1 > ct) or (bro1)) and ((tim2 <= ct)and(not bro2)):
        action = 4
    elif ((tim0 > ct) or (bro0)) and ((tim1 <= ct)and(not bro1)) and ((tim2 <= ct)and(not bro2)):
        action = 5
    elif ((tim0 <= ct)and(not bro0)) and ((tim1 > ct) or (bro1)) and ((tim2 > ct) or (bro2)):
        action = 0
    elif ((tim0 > ct) or (bro0)) and ((tim1 <= ct)and(not bro1)) and ((tim2 > ct) or (bro2)):
        action = 1
    elif ((tim0 > ct) or (bro0)) and ((tim1 > ct) or (bro1)) and ((tim2 <= ct)and(not bro2)):
        action = 2
    elif ((tim0 > ct) or (bro0)) and ((tim1 > ct) or (bro1)) and ((tim2 > ct) or (bro2)):
        action = np.argmin([tim0,tim1,tim2])
    return action

def set_manual_seed(manual_seed):
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

NUM_EPISODES = 1

job_shop = jobShop()

observations = []
actions =[]
rewards = []
MTTF = 534.6
ratio = 3
for episode in range(NUM_EPISODES):
    set_manual_seed(episode)
    done = False
    setting = random.randint(0, 2)
    inf = Info(setting,True,0)
    observation = job_shop.reset(inf)

    total_reward = 0.0
    episode_length = 0
    total_loss = 0.0
    while not done:
        observations.append(observation)
        action = RepairAgent(observation,MTTF,ratio)
        actions.append(action)
        observation, done, reward = job_shop.step(action)
        rewards.append(reward)
    mean_tardiness = mean_calculator(job_shop.tardiness,job_shop.tardiness,-1)
    mean_reward = mean_calculator(job_shop.rewards,job_shop.tardiness,-1)
    mean_flowtime = mean_calculator(job_shop.flowtime,job_shop.tardiness,0)

obs = pd.DataFrame(data=observations)
act = pd.DataFrame(data=actions)
rew = pd.DataFrame(data=rewards)
tard = pd.DataFrame(data=job_shop.tardiness)

obs.to_excel("Observations.xlsx")
act.to_excel("Actions.xlsx")
rew.to_excel("Rewards.xlsx")
tard.to_excel("Tardiness.xlsx")










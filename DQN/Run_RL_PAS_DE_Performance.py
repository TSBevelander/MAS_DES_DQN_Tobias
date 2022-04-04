import datetime
import random
import os
import pandas as pd

from make_environment import make_env
from agents import RandomAgent
import numpy as np
from torch.cuda import is_available
from RL_PAS_Dynamic_Events import jobShop, Info, get_objectives
from statistics import mean
from sklearn.linear_model import LinearRegression
from joblib import dump


import torch
observations = []
actions = []
tardiness = []
Results = []
Results2 = []
rewards = []
DTI = []
DTFO = []
DTTO = []


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

def RepairAgent1(observation,MTTF,ratio):
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

def RepairAgent2(observation):
    M1 = True
    M2 = True
    M3 = True
    TLR = 4000
    IAT = 2.5
    RTUF = 13
    IATobs = observation[56]
    RTUF1 = observation[42]
    RTUF2 = observation[47]
    RTUF3 = observation[52]
    TLR1 = observation[44]
    TLR2 = observation[49]
    TLR3 = observation[54]
    BRO1 = observation[45]
    BRO2 = observation[50]
    BRO3 = observation[55]
    if BRO1 and RTUF1 > RTUF:
        M1 = False
    if BRO2 and RTUF2 > RTUF:
        M2 = False
    if BRO3 and RTUF3 > RTUF:
        M3 = False
    if M1 and M2 and M3:
        action = 6
    elif not M1 and M2 and M3:
        action = 5
    elif M1 and not M2 and M3:
        action = 4
    elif M1 and M2 and not M3:
        action = 3
    elif not M1 and not M2 and M3:
        action = 2
    elif not M1 and M2 and not M3:
        action = 1
    elif M1 and not M2 and not M3:
        action = 0
    elif not M1 and not M2 and not M3:
        action = 6
    return action


def set_manual_seed(manual_seed):
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)

NUM_EPISODES = 1

job_shop = jobShop()

for setting in range(1):
    for episode in range(NUM_EPISODES):
        set_manual_seed(episode)
        done = False
        inf = Info(setting,True,0)
        observation = job_shop.reset(inf)
        while not done:
            observations.append(observation)
            action = AllAgent(observation)
            actions.append(action)
            observation, done, reward = job_shop.step(action)
        ms, ft, mt, mxt, t1, t2, t3, mw, et = get_objectives(job_shop,inf.min_job,inf.max_job,job_shop.early_termination)
        print("mean tardiness: " + str(mt) + ", mean flowtime: " + str(ft))
        Results.append([setting, episode, ms, ft, mt, mxt, t1, t2, t3, mw, et])
        print("end of setting: "+str(setting)+", episode: "+str(episode))

rew = pd.DataFrame(data=job_shop.rewards)
tard = pd.DataFrame(data=job_shop.tardiness)
rFlt = pd.DataFrame(data=job_shop.running_flowtime)





# for setting in range(1):
#         for episode in range(NUM_EPISODES):
#             set_manual_seed(episode)
#             done = False
#             inf = Info(setting,True,0)
#             observation = job_shop.reset(inf)
#             while not done:
#                 observations.append(observation)
#                 action = RepairAgent2(observation)
#                 actions.append(action)
#                 observation, done, reward = job_shop.step(action)
#             ms, ft, mt, mxt, t1, t2, t3, mw, et = get_objectives(job_shop,inf.min_job,inf.max_job,job_shop.early_termination)
#             print("mean tardiness: "+str(mt)+", mean flowtime: "+str(ft))
#             Results.append([setting, episode, ms, ft, mt, mxt, t1, t2, t3, mw, et])
#             print("end of setting: "+str(setting)+", episode: "+ str(episode))


# Res = pd.DataFrame(data=Results)
Flt = pd.DataFrame(data=job_shop.flowtime)
# rFlt.to_excel("RunningFlowtime.xlsx")
#
# Res.to_excel("Results.xlsx")
# obs = pd.DataFrame(data=observations)
# act = pd.DataFrame(data=actions)
# # rew = pd.DataFrame(data=rewards)
tard = pd.DataFrame(data=job_shop.tardiness)
# #

mach = pd.DataFrame(data=job_shop.machinelog)
# cfpt = pd.DataFrame(data=cfptime)
# obs.to_excel("Observations.xlsx")
# act.to_excel("Actions.xlsx")
rew.to_excel("Rewards.xlsx")
tard.to_excel("Tardiness.xlsx")
# mach.to_excel("MachineLog.xlsx")
# # cfpt.to_excel("CfpTimes.xlsx")
# Flt.to_excel("Flowtime.xlsx")











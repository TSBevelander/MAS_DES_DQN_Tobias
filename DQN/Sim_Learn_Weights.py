"""
DES of the MAS. This DES is used to optimize certain aspects of the
MAS such as the bids. It can be used to quickly run multiple experiments.
The results can then be implemented in the MAS to see the effects.
"""
import csv
import math
import random
import sys
from collections import defaultdict
from functools import partial

import numpy as np
# import matplotlib.cbook
import pandas as pd
import simpy
from pathos.multiprocessing import ProcessingPool as Pool
from simpy import *
from RL_PAS_Dynamic_Events import jobShop, Info
from Run_RL_PAS import AllAgent

# warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

number = 2500  # Max number of jobs if infinite is false
noJobCap = True  # For infinite
maxTime = 10000.0  # Runtime limit
# Machine shop settings
processingTimes = [[6.75], [3.75], [7.5]]
operationOrder = [[1], [1], [1]]
numberOfOperations = [1, 1, 1]
machinesPerWC = [3]
machine_number_WC = [[1, 2, 3]]
setupTime = [[0, 0.625, 1.25], [0.625, 0, 0.8], [1.25, 0.8, 0]]
mean_setup = [0.515, 0.306, 0.515, 0.429, 0.306]
demand = [0.3, 0.5, 0.2]
noOfWC = range(len(machinesPerWC))

# arrivalMean = 1.4572
# Mean of arrival process
# dueDateTightness = 3

if noJobCap:
    number = 0

"Initial parameters of the GES"
noAttributes = 8
noAttributesJob = 4
totalAttributes = noAttributes + noAttributesJob

no_generation = 1000

def do_simulation_with_weights(mean_weight_new, std_weight_new, arrivalMean, due_date_tightness, bid_skip, seq_skip,
                               norm_range, min_job, max_job, wip_max,uti,iii):
    eta_new = np.zeros((sum(machinesPerWC), totalAttributes))
    objective_new = np.zeros(2)
    mean_tard = np.zeros(2)
    max_tard = np.zeros(2)
    test_weights_pos = np.zeros((sum(machinesPerWC), totalAttributes))
    test_weights_min = np.zeros((sum(machinesPerWC), totalAttributes))
    for mm in range(sum(machinesPerWC)):
        for jj in range(totalAttributes):
            if (jj == noAttributes - 1) | (jj == noAttributesJob + noAttributes - 1) | (jj in bid_skip) | (
                    jj in [x + noAttributes for x in seq_skip]):
                eta_new[mm][jj] = 0
                test_weights_pos[mm][jj] = 0
                test_weights_min[mm][jj] = 0
            else:
                eta_new[mm][jj] = random.gauss(0, np.exp(std_weight_new[mm][jj]))
                test_weights_pos[mm][jj] = mean_weight_new[mm][jj] + (eta_new[mm][jj])
                test_weights_min[mm][jj] = mean_weight_new[mm][jj] - (eta_new[mm][jj])

    job_shop = jobShop()
    done = False
    setting = uti
    inf = Info(setting,False,test_weights_pos)
    observation = job_shop.reset(inf)

    while not done:
        action = AllAgent(observation)
        observation, done, reward = job_shop.step(action)

    if job_shop.early_termination == 1:
        if math.isnan(np.nanmean(np.nonzero(job_shop.tardiness[min_job:max_job]))):
            objective_new[0] = 20_000
            max_tard[0] = 1000
        else:
            objective_new[0] = np.nanmean(np.nonzero(job_shop.tardiness[min_job:max_job])) + 10_000 - np.count_nonzero(
                job_shop.flowtime[min_job:max_job]) + 0.01 * max(job_shop.tardiness[min_job:max_job])
            max_tard[0] = np.nanmax(job_shop.tardiness[min_job:max_job])
    else:
        objective_new[0] = np.nanmean(job_shop.tardiness[min_job:max_job]) + 0.01 * max(
            job_shop.tardiness[min_job:max_job])
        max_tard[0] = np.nanmax(job_shop.tardiness[min_job:max_job])
        # print(job_shop.tardiness[499:2499])

    mean_tard[0] = np.nanmean(job_shop.tardiness[min_job:max_job])
    # max_tard[0] = max(job_shop.tardiness[min_job:max_job])

    # seed = random.randrange(sys.maxsize)
    # random.seed(seed)
    # print("Seed was:", seed)
    job_shop = jobShop()
    done = False
    setting = uti
    inf = Info(setting, False, test_weights_min)
    observation = job_shop.reset(inf)

    while not done:
        action = AllAgent(observation)
        observation, done, reward = job_shop.step(action)

    if job_shop.early_termination == 1:
        if math.isnan(np.nanmean(np.nonzero(job_shop.tardiness[min_job:max_job]))):
            objective_new[1] = 20_000
            max_tard[1] = 1000
        else:
            objective_new[1] = np.nanmean(np.nonzero(job_shop.tardiness[min_job:max_job])) + 10_000 - np.count_nonzero(
                job_shop.flowtime[min_job:max_job]) + 0.01 * max(job_shop.tardiness[min_job:max_job])
            max_tard[1] = np.nanmax(job_shop.tardiness[min_job:max_job])
    else:
        objective_new[1] = np.nanmean(job_shop.tardiness[min_job:max_job]) + 0.01 * max(
            job_shop.tardiness[min_job:max_job])
        max_tard[1] = np.nanmax(job_shop.tardiness[min_job:max_job])
    # print(len(job_shop.tardiness))

    mean_tard[1] = np.nanmean(job_shop.tardiness[min_job:max_job])
    # max_tard[1] = np.nanmax(job_shop.tardiness[min_job:max_job])

    return objective_new, eta_new, mean_tard, max_tard


def run_linear(filename1, filename2, arrival_time_mean, due_date_k, alpha, bid_skip, seq_skip, norm_range, min_job,
               max_job, wip_max, uti):
    # str1 = "Runs/Final_runs/Run-weights-85-6.csv"
    # df = pd.read_csv(str1, header=None)
    # mean_weight = df.values.tolist()
    file1 = open(filename1, "w")
    mean_weight = np.zeros((sum(machinesPerWC), totalAttributes))
    std_weight = np.zeros((sum(machinesPerWC), totalAttributes))
    for m in range(sum(machinesPerWC)):
        for j in range(totalAttributes):
            if (j == noAttributes - 1) | (j == noAttributesJob + noAttributes - 1) | (j in bid_skip) | (
                    j in [x + noAttributes for x in seq_skip]):
                std_weight[m][j] = 0
            else:
                std_weight[m][j] = std_weight[m][j] + np.log(0.3)
    population_size = totalAttributes + 1

    for i in range(sum(machinesPerWC)):
        mean_weight[i][0] = -0.5
        mean_weight[i][6] = -3

        mean_weight[i][noAttributes] = -1
        mean_weight[i][noAttributes + 2] = -3

    jobshop_pool = Pool(processes=population_size)
    alpha_mean = 0.1
    alpha_std = 0.025
    beta_1 = 0.9
    beta_2 = 0.999
    m_t_mean = np.zeros((sum(machinesPerWC), totalAttributes))
    v_t_mean = np.zeros((sum(machinesPerWC), totalAttributes))

    m_t_std = np.zeros((sum(machinesPerWC), totalAttributes))
    v_t_std = np.zeros((sum(machinesPerWC), totalAttributes))

    objective = np.zeros((population_size, 2))
    eta = np.zeros((population_size, sum(machinesPerWC), totalAttributes))
    mean_tardiness = np.zeros((population_size, 2))
    max_tardiness = np.zeros((population_size, 2))

    for num_sim in range(no_generation):
        seeds = range(int(population_size))
        func1 = partial(do_simulation_with_weights, mean_weight, std_weight, arrival_time_mean, due_date_k,
                        bid_skip, seq_skip, norm_range, min_job, max_job, wip_max, uti)
        makespan_per_seed = jobshop_pool.map(func1, seeds)
        for h, j in zip(range(int(population_size)), seeds):
            objective[j] = makespan_per_seed[h][0]
            eta[j] = makespan_per_seed[h][1]
            mean_tardiness[j] = makespan_per_seed[h][2]
            max_tardiness[j] = makespan_per_seed[h][3]

        objective_norm = np.zeros((population_size, 2))
        # Normalise the current populations performance
        for ii in range(population_size):
            objective_norm[ii][0] = (objective[ii][0] - np.mean(objective, axis=0)[0]) / np.std(objective, axis=0)[0]

            objective_norm[ii][1] = (objective[ii][1] - np.mean(objective, axis=0)[1]) / np.std(objective, axis=0)[1]

        delta_mean_final = np.zeros((sum(machinesPerWC), totalAttributes))
        delta_std_final = np.zeros((sum(machinesPerWC), totalAttributes))
        for m in range(sum(machinesPerWC)):
            for j in range(totalAttributes):
                delta_mean = 0
                delta_std = 0
                for ii in range(population_size):
                    delta_mean += ((objective_norm[ii][0] - objective_norm[ii][1]) / 2) * eta[ii][m][j] / np.exp(
                        std_weight[m][j])

                    delta_std += ((objective_norm[ii][0] + objective_norm[ii][1]) / 2) * (eta[ii][m][j] ** 2 - np.exp(
                        std_weight[m][j])) / (np.exp(std_weight[m][j]))

                delta_mean_final[m][j] = delta_mean / population_size
                delta_std_final[m][j] = delta_std / population_size

        # print(delta_std_final)
        t = num_sim + 1
        for m in range(sum(machinesPerWC)):
            for j in range(totalAttributes):
                m_t_mean[m][j] = (beta_1 * m_t_mean[m][j] + (1 - beta_1) * delta_mean_final[m][j])
                v_t_mean[m][j] = (beta_2 * v_t_mean[m][j] + (1 - beta_2) * delta_mean_final[m][j] ** 2)
                m_hat_t = (m_t_mean[m][j] / (1 - beta_1 ** t))
                v_hat_t = (v_t_mean[m][j] / (1 - beta_2 ** t))
                mean_weight[m][j] = mean_weight[m][j] - (alpha_mean * m_hat_t) / (np.sqrt(v_hat_t) + 10 ** -8)

                m_t_std[m][j] = (beta_1 * m_t_std[m][j] + (1 - beta_1) * delta_std_final[m][j])
                v_t_std[m][j] = (beta_2 * v_t_std[m][j] + (1 - beta_2) * delta_std_final[m][j] ** 2)
                m_hat_t_1 = (m_t_std[m][j] / (1 - beta_1 ** t))
                v_hat_t_1 = (v_t_std[m][j] / (1 - beta_2 ** t))
                std_weight[m][j] = std_weight[m][j] - (alpha_std * m_hat_t_1) / (np.sqrt(v_hat_t_1) + 10 ** -8)

        alpha_mean = 0.1 * np.exp(-(t - 1) / alpha)
        alpha_std = 0.025 * np.exp(-(t - 1) / alpha)

        #
        # final_objective.append(np.mean(np.mean(objective, axis=0)))
        # Ln.set_ydata(final_objective)
        # Ln.set_xdata(range(len(final_objective)))
        # plt.pause(1)

        # print(objective)

        objective1 = np.array(objective)

        # print(num_sim, objective1[objective1 < 5000].mean(), np.mean(np.mean(np.exp(std_weight))))
        print(num_sim, np.nanmean(mean_tardiness), np.nanmean(max_tardiness))
        # print(np.mean(np.exp(std_weight), axis=0))
        L = [str(num_sim) + " ", str(np.mean(np.mean(objective, axis=0))) + " ",
             str(np.mean(np.exp(std_weight), axis=0)) + "\n"]
        file1.writelines(L)
        if num_sim == no_generation - 1:
            print(np.exp(std_weight))
            # file1.close()
            file2 = open(filename2 + ".csv", 'w')
            writer = csv.writer(file2)
            writer.writerows(mean_weight)
            file2.close()


if __name__ == '__main__':
    arrival_time = [2.118, 2.0, 1.895]
    utilization = [85, 90, 95]
    due_date_settings = [4, 4, 4]
    learning_decay_rate = [10, 100, 500, 1000, 2500, 5000, 10000]
    # att_considered = [10, 9, 9, 9, 9, 9, 9, 9, 9, 9]

    normaliziation = [[14, 30, 3.5, 4, -600, 23],[14, 30, 3.5, 4, -600, 23],[14, 30, 3.5, 4, -600, 23]]

    min_jobs = [2499, 2499, 2499]
    max_jobs = [4499, 4499, 4499]
    wip_max = [50, 50, 150]

    # skip_bid = [[0, 7], [1, 7], [3, 7], [5, 7], [7, 7], [7, 7], [7, 7]]
    # skip_seq = [[3, 3], [3, 3], [3, 3], [3, 3], [0, 3], [1, 3], [2, 3]]
    #
    skip_bid = [[7, 7], [2, 7], [4, 7], [5, 7]]
    skip_seq = [[3, 3], [3, 3], [3, 3], [3, 3]]

    for a in range(2,3):
        for skip in range(1):
            for n in range(1):
                print("Current run is:" + str(utilization[a]) + "-" + str(due_date_settings[a]) + "-" + str(
                    learning_decay_rate[3]) + "-" + str(skip_bid[skip]) + "-" + str(skip_seq[skip]))
                str1 = "DQNRuns/Attribute_Runs/" + str(utilization[a]) + "-" + str(
                    due_date_settings[a]) + "/Run-" + str(utilization[a]) + "-" + str(
                    due_date_settings[a]) + "-" + str(str(learning_decay_rate[3])) + "-" + str(skip_bid[skip]) + "-" + str(
                    skip_seq[skip]) + ".txt"
                str2 = "DQNRuns/Attribute_Runs/" + str(utilization[a]) + "-" + str(
                    due_date_settings[a]) + "/Run-weights-" + str(utilization[a]) + "-" + str(
                    due_date_settings[a]) + "-" + str(learning_decay_rate[3]) + "-" + str(skip_bid[skip]) + "-" + str(
                    skip_seq[skip])
                run_linear(str1, str2, arrival_time[a], due_date_settings[a], learning_decay_rate[3], skip_bid[skip],
                           skip_seq[skip], normaliziation[a], min_jobs[a], max_jobs[a], wip_max[a], a)



"""
DES of the MAS. This DES is used to optimize certain aspects of the
MAS such as the bids. It can be used to quickly run multiple experiments.
The results can then be implemented in the MAS to see the effects.
"""
import csv
import random
import sys
from collections import defaultdict
from functools import partial

import numpy as np
# import matplotlib.cbook
import simpy
from pathos.multiprocessing import ProcessingPool as Pool
from simpy import *

# warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

number = 2500  # Max number of jobs if infinite is false
noJobCap = True  # For infinite
maxTime = 10000.0  # Runtime limit
processingTimes = [[6.75, 3.75, 2.5, 7.5], [3.75, 5.0, 7.5], [3.75, 2.5, 8.75, 5.0, 5.0]]
operationOrder = [[3, 1, 2, 5], [4, 1, 3], [2, 5, 1, 4, 3]]
numberOfOperations = [4, 3, 5]
machinesPerWC = [4, 2, 5, 3, 2]
machine_number_WC = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11], [12, 13, 14], [15, 16]]
setupTime = [[0, 0.625, 1.25], [0.625, 0, 0.8], [1.25, 0.8, 0]]
mean_setup = [0.515, 0.306, 0.515, 0.429, 0.306]
demand = [0.2, 0.5, 0.3]


if noJobCap:
    number = 0

"Initial parameters of the GES"
noAttributes = 8
noAttributesJob = 3
totalAttributes = noAttributes + noAttributesJob

no_generation = 500


def list_duplicates(seq):
    tally = defaultdict(list)
    for ii, item in enumerate(seq):
        tally[item].append(ii)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) >= 1)


def bid_winner(env, job, noOfMachines, currentWC, job_shop, last_job, makespan_currentWC, machine, store):
    current_bid = [0] * noOfMachines
    current_job = [0] * noOfMachines
    best_bid = []
    best_job = []
    no_of_jobs = len(job)
    total_rp = [0] * no_of_jobs
    for j in range(no_of_jobs):
        total_rp[j] = (remain_processing_time(job[j]))

    for jj in range(noOfMachines):
        queue_length = len(machine[jj].items)
        # expected_start = expected_start_time(jj, currentWC, machine)
        # start_time = max(env.now, makespan_currentWC[jj] + expected_start)

        new_bid = [0] * no_of_jobs
        i = 0
        for j in job:
            if last_job[jj] == 0:
                min_setup_time = 0
            else:
                min_setup_time = minimum_setup_time(machine[jj].items, j, last_job[jj])

            attributes = bid_calculation(job_shop.test_weights, machine_number_WC[currentWC - 1][jj],
                                         j.processingTime[j.currentOperation - 1], j.currentOperation,
                                         j.numberOfOperations,
                                         j.dueDate[j.currentOperation], total_rp[i], queue_length,
                                         j.dueDate[j.numberOfOperations], env.now,
                                         j.priority, min_setup_time)
            new_bid[i] = attributes
            i += 1

        ind_winning_job = new_bid.index(max(new_bid))
        current_bid[jj] = new_bid[ind_winning_job]
        current_job[jj] = ind_winning_job

    # Determine the winning bids
    sorted_list = sorted(list_duplicates(current_job))
    for dup in sorted_list:
        bestmachine = dup[1][0]
        bestbid = current_bid[bestmachine]

        for ii in dup[1]:
            if bestbid <= current_bid[ii]:
                bestbid = current_bid[ii]
                bestmachine = ii

        best_bid.append(bestmachine)  # Machine winner
        best_job.append(int(dup[0]))  # Job to be processed

    for ii in range(len(best_job)):
        put_job_in_queue(currentWC, best_bid[ii], job[best_job[ii]], job_shop, env, machine)

    for ii in reversed(best_job):
        yield store.get(lambda mm: mm == job[ii])


def remain_processing_time(job):
    total_rp = 0
    for ii in range(job.currentOperation - 1, job.numberOfOperations):
        total_rp += job.processingTime[ii]

    return total_rp


def minimum_setup_time(pool, job, last_job):
    min_setup = [setupTime[job.type - 1][int(last_job) - 1]]
    for jj in pool:
        min_setup.append(setupTime[job.type - 1][jj.type - 1])

    return min(min_setup)


def next_workstation(job, job_shop, env, all_store):
    if job.currentOperation + 1 <= job.numberOfOperations:
        job.currentOperation += 1
        nextWC = operationOrder[job.type - 1][job.currentOperation - 1]
        store = all_store[nextWC - 1]
        store.put(job)
    else:
        finish_time = env.now
        job_shop.tardiness[job.number] = max(job.priority * (finish_time - job.dueDate[job.numberOfOperations]), 0)
        job_shop.WIP -= 1
        job_shop.flowtime[job.number] = finish_time - job.dueDate[0]

        if job.number > 2499:
            if np.count_nonzero(job_shop.flowtime[499:2499]) == 2000:
                job_shop.finishtime = env.now
                job_shop.end_event.succeed()

        if (job_shop.WIP > 3500) | (env.now > 75_000):
            # print(job_shop.WIP)
            job_shop.end_event.succeed()
            job_shop.early_termination = 1


def bid_calculation(weights_new, machinenumber, processing_time,
                    current, total, due_date_operation, total_rp, queue_length, due_date, now, job_priority, min_setup_time):
    attribute = [0] * noAttributes

    attribute[0] = processing_time / 8.75 * weights_new[machinenumber - 1][0]
    attribute[1] = current / total * weights_new[machinenumber - 1][1]
    attribute[2] = (due_date_operation - now + 400) / (97.5 + 400) * \
                   weights_new[machinenumber - 1][2]
    attribute[3] = total_rp / 21.25 * weights_new[machinenumber - 1][3]
    attribute[4] = (((due_date - now) / total_rp) + 7) / (12 + 7) * weights_new[machinenumber - 1][4]  # Critical Ratio
    attribute[5] = (job_priority - 1) / (10 - 1) * weights_new[machinenumber - 1][5]  # Job Weight
    attribute[6] = queue_length / 10 * weights_new[machinenumber - 1][6]  # Current Workload of Machine
    attribute[7] = min_setup_time / 1.25 * weights_new[machinenumber - 1][7]  # minimum setup time

    return sum(attribute)


def set_makespan(current_makespan, job, last_job, env, setup_time):
    add = current_makespan + job.processingTime[job.currentOperation - 1] + setup_time

    new = env.now + job.processingTime[job.currentOperation - 1] + setup_time

    return max(add, new)


def put_job_in_queue(currentWC, choice, job, job_shop, env, machines):
    machines[choice].put(job)
    if not job_shop.condition_flag[currentWC - 1][choice].triggered:
        job_shop.condition_flag[currentWC - 1][choice].succeed()


def choose_job_queue(weights_new_job, machinenumber, processing_time, due_date, env,
                     setup_time,
                     job_priority):
    attribute_job = [0] * noAttributesJob

    attribute_job[2] = setup_time / 1.25 * weights_new_job[machinenumber - 1][noAttributes + 2]
    attribute_job[1] = (job_priority - 1) / (10 - 1) * weights_new_job[machinenumber - 1][noAttributes + 1]
    attribute_job[0] = (due_date - processing_time - setup_time - env.now - (-400)) / (100 + 400) * \
                       weights_new_job[machinenumber - 1][noAttributes]

    # attribute_job[2] = normalize(setup_time, 1.25, 0) * weights_new_job[machinenumber - 1][noAttributes + 2]
    # attribute_job[1] = normalize(job_priority, 10, 1) * weights_new_job[machinenumber - 1][noAttributes + 1]
    # attribute_job[0] = normalize(due_date - processing_time - setup_time - env.now, 100, -400) * \
    #                    weights_new_job[machinenumber - 1][noAttributes]

    # total_bid = sum(attribute_job)
    return sum(attribute_job)


def machine_processing(job_shop, current_WC, machine_number, env, weights_new, relative_machine, last_job, machine,
                       makespan, all_store):
    while True:
        relative_machine = machine_number_WC[current_WC - 1].index(machine_number)
        if machine[relative_machine].items:
            setup_time = []
            priority_list = []
            if (len(machine[relative_machine].items) == 1) | (last_job[relative_machine] == 0):
                ind_processing_job = 0
                setup_time.append(0)
            else:
                for job in machine[relative_machine].items:
                    setuptime = setupTime[job.type - 1][int(last_job[relative_machine]) - 1]
                    # priority_list.append(job.dueDate[job.numberOfOperations])
                    job_queue_priority = choose_job_queue(weights_new, machine_number,
                                                          job.processingTime[job.currentOperation - 1],
                                                          job.dueDate[job.numberOfOperations], env, setuptime,
                                                          job.priority)
                    priority_list.append(job_queue_priority)
                    setup_time.append(setuptime)
                ind_processing_job = priority_list.index(max(priority_list))

            # ind_processing_job = 0
            next_job = machine[relative_machine].items[ind_processing_job]
            setuptime = setup_time[ind_processing_job]
            tip = next_job.processingTime[next_job.currentOperation - 1] + setuptime
            makespan[relative_machine] = set_makespan(makespan[relative_machine], next_job, last_job[relative_machine],
                                                      env, setuptime)
            last_job[relative_machine] = next_job.type
            # schedule.append(
            #     [relative_machine, next_job.name, next_job.type, makespan[relative_machine], tip, setup_time, env.now,
            #      next_job.priority])
            machine[relative_machine].items.remove(next_job)
            yield env.timeout(tip)
            next_workstation(next_job, job_shop, env, all_store)
        else:
            yield job_shop.condition_flag[current_WC - 1][relative_machine]
            job_shop.condition_flag[current_WC - 1][relative_machine] = simpy.Event(env)


def cfp_wc(env, last_job, machine, makespan, store, job_shop, currentWC):
    while True:
        if store.items:
            c = bid_winner(env, store.items, machinesPerWC[currentWC], currentWC + 1, job_shop, last_job, makespan,
                           machine, store)
            env.process(c)
        tib = 0.5
        yield env.timeout(tib)


def no_in_system(R):
    """Total number of jobs in the resource R"""
    return len(R.put_queue) + len(R.users)


def source(env, number1, interval, job_shop, due_date_setting):
    if not noJobCap:  # If there is a limit on the number of jobs
        for ii in range(number1):
            job = New_Job('job%02d' % ii, env, ii, due_date_setting)
            firstWC = operationOrder[job.type - 1][0]
            store = eval('job_shop.storeWC' + str(firstWC))
            store.put(job)
            # d = job_pool_agent(job, firstWC, job_shop, store)
            # env.process(d)
            tib = random.expovariate(1.0 / interval)
            yield env.timeout(tib)
    else:
        while True:  # Needed for infinite case as True refers to "until".
            ii = number1
            number1 += 1
            job = New_Job('job%02d' % ii, env, ii, due_date_setting)
            job_shop.tardiness.append(-1)
            job_shop.flowtime.append(0)
            job_shop.WIP += 1
            firstWC = operationOrder[job.type - 1][0]
            store = eval('job_shop.storeWC' + str(firstWC))
            store.put(job)
            # d = job_pool_agent(job, firstWC, job_shop, store)
            # env.process(d)
            tib = random.expovariate(1.0 / interval)
            # tib = random.uniform(0.8, 1.2)
            yield env.timeout(tib)


class jobShop:
    def __init__(self, env, weights):
        machine_wc1 = {ii: Store(env) for ii in range(machinesPerWC[0])}
        machine_wc2 = {ii: Store(env) for ii in range(machinesPerWC[1])}
        machine_wc3 = {ii: Store(env) for ii in range(machinesPerWC[2])}
        machine_wc4 = {ii: Store(env) for ii in range(machinesPerWC[3])}
        machine_wc5 = {ii: Store(env) for ii in range(machinesPerWC[4])}

        job_poolwc1 = simpy.FilterStore(env)
        job_poolwc2 = simpy.FilterStore(env)
        job_poolwc3 = simpy.FilterStore(env)
        job_poolwc4 = simpy.FilterStore(env)
        job_poolwc5 = simpy.FilterStore(env)

        self.machinesWC1 = machine_wc1
        self.machinesWC2 = machine_wc2
        self.machinesWC3 = machine_wc3
        self.machinesWC4 = machine_wc4
        self.machinesWC5 = machine_wc5

        self.QueuesWC1 = []
        self.QueuesWC2 = []
        self.QueuesWC3 = []
        self.QueuesWC4 = []
        self.QueuesWC5 = []

        self.scheduleWC1 = []
        self.scheduleWC2 = []
        self.scheduleWC3 = []
        self.scheduleWC4 = []
        self.scheduleWC5 = []

        self.condition_flag = []
        for wc in range(len(machinesPerWC)):
            Q = eval('self.QueuesWC' + str(wc + 1))
            self.condition_flag.append([])
            for ii in range(machinesPerWC[wc]):
                Q.append([])
                self.condition_flag[wc].append(simpy.Event(env))

        self.makespanWC1 = np.zeros(machinesPerWC[0])
        self.makespanWC2 = np.zeros(machinesPerWC[1])
        self.makespanWC3 = np.zeros(machinesPerWC[2])
        self.makespanWC4 = np.zeros(machinesPerWC[3])
        self.makespanWC5 = np.zeros(machinesPerWC[4])

        self.last_job_WC1 = np.zeros(machinesPerWC[0])
        self.last_job_WC2 = np.zeros(machinesPerWC[1])
        self.last_job_WC3 = np.zeros(machinesPerWC[2])
        self.last_job_WC4 = np.zeros(machinesPerWC[3])
        self.last_job_WC5 = np.zeros(machinesPerWC[4])

        self.storeWC1 = job_poolwc1
        self.storeWC2 = job_poolwc2
        self.storeWC3 = job_poolwc3
        self.storeWC4 = job_poolwc4
        self.storeWC5 = job_poolwc5

        self.test_weights = weights
        self.makespan = []
        self.tardiness = []
        self.WIP = 0
        self.early_termination = 0

        self.bids = []


class New_Job:
    def __init__(self, name, env, number1, dueDateTightness):
        jobType = random.choices([1, 2, 3], weights=demand, k=1)
        jobWeight = random.choices([1, 3, 10], weights=[0.5, 0.3, 0.2], k=1)
        self.type = jobType[0]
        self.priority = jobWeight[0]
        self.number = number1
        self.name = name
        self.currentOperation = 1
        self.processingTime = np.zeros(numberOfOperations[self.type - 1])
        self.dueDate = np.zeros(numberOfOperations[self.type - 1] + 1)
        self.dueDate[0] = env.now
        self.operationOrder = operationOrder[self.type - 1]
        self.numberOfOperations = numberOfOperations[self.type - 1]
        for ii in range(self.numberOfOperations):
            meanPT = processingTimes[self.type - 1][ii]
            self.processingTime[ii] = meanPT
            self.dueDate[ii + 1] = self.dueDate[ii] + self.processingTime[ii] * dueDateTightness


def do_simulation_with_weights(mean_weight_new, std_weight_new, arrivalMean, due_date_tightness, iii):
    eta_new = np.zeros((sum(machinesPerWC), totalAttributes))
    objective_new = np.zeros(2)
    test_weights_pos = np.zeros((sum(machinesPerWC), totalAttributes))
    test_weights_min = np.zeros((sum(machinesPerWC), totalAttributes))
    for mm in range(sum(machinesPerWC)):
        for jj in range(totalAttributes):
            eta_new[mm][jj] = random.normalvariate(0, np.exp(std_weight_new[mm][jj]))
            test_weights_pos[mm][jj] = mean_weight_new[mm][jj] + (eta_new[mm][jj])
            test_weights_min[mm][jj] = mean_weight_new[mm][jj] - (eta_new[mm][jj])

    # seed = random.randrange(sys.maxsize)
    # random.seed(seed)
    # print("Seed was:", seed)
    env = Environment()
    job_shop = jobShop(env, test_weights_pos)
    env.process(source(env, number, arrivalMean, job_shop, due_date_tightness))
    all_stores = []

    for wc in range(len(machinesPerWC)):
        last_job = eval('job_shop.last_job_WC' + str(wc + 1))
        machine = eval('job_shop.machinesWC' + str(wc + 1))
        makespan = eval('job_shop.makespanWC' + str(wc + 1))
        store = eval('job_shop.storeWC' + str(wc + 1))
        all_stores.append(store)

        env.process(cfp_wc(env, last_job, machine, makespan, store, job_shop, wc))

        for ii in range(machinesPerWC[wc]):
            env.process(
                machine_processing(job_shop, wc + 1, machine_number_WC[wc][ii], env, test_weights_pos, ii, last_job,
                                   machine, makespan, all_stores))

    job_shop.end_event = env.event()

    env.run(until=job_shop.end_event)
    if job_shop.early_termination == 1:
        objective_new[0] = np.mean(np.nonzero(job_shop.tardiness[499:2999])) + 10_000 - np.count_nonzero(
            job_shop.makespan[499:2499])
    else:
        objective_new[0] = np.mean(job_shop.tardiness[499:2499])

    # seed = random.randrange(sys.maxsize)
    # random.seed(seed)
    # print("Seed was:", seed)
    env = Environment()
    job_shop = jobShop(env, test_weights_min)
    env.process(source(env, number, arrivalMean, job_shop, due_date_tightness))
    all_stores = []

    for wc in range(len(machinesPerWC)):
        last_job = eval('job_shop.last_job_WC' + str(wc + 1))
        machine = eval('job_shop.machinesWC' + str(wc + 1))
        makespan = eval('job_shop.makespanWC' + str(wc + 1))
        store = eval('job_shop.storeWC' + str(wc + 1))
        all_stores.append(store)

        env.process(cfp_wc(env, last_job, machine, makespan, store, job_shop, wc))

        for ii in range(machinesPerWC[wc]):
            env.process(
                machine_processing(job_shop, wc + 1, machine_number_WC[wc][ii], env, test_weights_min, ii, last_job,
                                   machine, makespan, all_stores))

    job_shop.end_event = env.event()

    env.run(until=job_shop.end_event)
    if job_shop.early_termination == 1:
        objective_new[1] = np.mean(np.nonzero(job_shop.tardiness[499:2499])) + 10_000 - np.count_nonzero(
            job_shop.makespan[499:2499])
    else:
        objective_new[1] = np.mean(job_shop.tardiness[499:2499])
    # print(len(job_shop.tardiness))

    return objective_new, eta_new


def run_linear(filename1, filename2, filename3, arrival_time_mean, due_date_k, alpha):
    file1 = open(filename1, "w")
    mean_weight = np.zeros((sum(machinesPerWC), totalAttributes))
    std_weight = np.zeros((sum(machinesPerWC), totalAttributes))
    for m in range(sum(machinesPerWC)):
        for j in range(totalAttributes):
            if j < 8:
                std_weight[m][j] = std_weight[m][j] + np.log(0.3)
            else:
                std_weight[m][j] = std_weight[m][j] + np.log(0.3)
    population_size = totalAttributes + 4

    for i in range(sum(machinesPerWC)):
        mean_weight[i][0] = -0.5
        # mean_weight[i][7] = -1
        # mean_weight[i][2] = -3
        mean_weight[i][6] = -3
        mean_weight[i][7] = -3

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
    for num_sim in range(no_generation):
        for ii in range(1):
            seeds = range(int(population_size))
            func1 = partial(do_simulation_with_weights, mean_weight, std_weight, arrival_time_mean, due_date_k)
            makespan_per_seed = jobshop_pool.map(func1, seeds)
            for h, j in zip(range(int(population_size)), seeds):
                objective[ii * int(population_size) + j] = makespan_per_seed[h][0]
                eta[ii * int(population_size) + j] = makespan_per_seed[h][1]

        # print(objective)

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

        # print(delta_std_final[0:15][11])

        # print(delta_std_final)
        t = num_sim + 1
        for m in range(sum(machinesPerWC)):
            for j in range(totalAttributes):
                m_t_mean[m][j] = (beta_1 * m_t_mean[m][j] + (1 - beta_1) * delta_mean_final[m][j])
                v_t_mean[m][j] = (beta_2 * v_t_mean[m][j] + (1 - beta_2) * delta_mean_final[m][j] ** 2)
                m_hat_t = (m_t_mean[m][j] / (1 - beta_1 ** t))
                v_hat_t = (v_t_mean[m][j] / (1 - beta_2 ** t))
                mean_weight[m][j] = mean_weight[m][j] - (alpha_mean * m_hat_t) / (np.sqrt(v_hat_t) + 10 ** -8)

                # std_weight[m][j] = std_weight[m][j] - alpha_std * delta_std_final[m][j]
                #
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

        print(num_sim, np.mean(np.mean(objective, axis=0)), np.mean(np.mean(np.exp(std_weight))))
        # print(np.mean(np.exp(std_weight), axis=0))
        L = [str(num_sim) + " ", str(np.mean(np.mean(objective, axis=0))) + " ",
             str(np.mean(np.exp(std_weight), axis=0)) + "\n"]
        file1.writelines(L)

        if num_sim == no_generation - 1:
            print(np.exp(std_weight))
            file1.close()
            file2 = open(filename2, 'w')
            writer = csv.writer(file2)
            writer.writerows(mean_weight)
            file2.close()

            file3 = open('filename3', 'w')
            writer = csv.writer(file3)
            writer.writerows(std_weight)
            file3.close()


if __name__ == '__main__':
    arrival_time = [1.5429, 1.5429, 1.5429, 1.4572, 1.4572, 1.4572, 1.3804, 1.3804, 1.3804]
    utilization = [85, 85, 85, 90, 90, 90, 95, 95, 95]
    due_date_settings = [4, 6, 8, 4, 6, 8, 4, 6, 8]
    learning_decay_rate = [10, 100, 500, 1000, 2500, 5000, 10000]
    for num_runs in range(3, 4):
        print("Current run is:" + str(utilization[num_runs]) + "-" + str(due_date_settings[num_runs]) + "-" + str(
            learning_decay_rate[5]))
        str1 = "Runs/Attribute_Runs/Run-All-" + str(utilization[num_runs]) + "-" + str(
            due_date_settings[num_runs]) + "-" + str(str(learning_decay_rate[5])) + ".txt"
        str2 = "Runs/Attribute_Runs/Run-Weights-All-" + str(utilization[num_runs]) + "-" + str(
            due_date_settings[num_runs]) + "-" + str(str(learning_decay_rate[5])) + ".csv"
        str3 = "Runs/Attribute_Runs/Run-Std-All-" + str(utilization[num_runs]) + "-" + str(
            due_date_settings[num_runs]) + "-" + str(str(learning_decay_rate[5])) + ".csv"
        run_linear(str1, str2, str3, arrival_time[num_runs], due_date_settings[num_runs], learning_decay_rate[5])

    # # to run GUI event loop
    # dat = [0, 1]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # Ln, = ax.plot(dat)
    # ax.set_xlim([0, 1000])
    # ax.set_ylim([0, 150])
    # plt.ion()
    # plt.show()
    #
    # # setting title
    # plt.title("Mean objective function", fontsize=20)
    #
    # # setting x-axis label and y-axis label
    # plt.xlabel("No. of iterations")
    # plt.ylabel("Obejctive Function")

"""
DES of the MAS. This DES is used to optimize certain aspects of the
MAS such as the bids. It can be used to quickly run multiple experiments.
The results can then be implemented in the MAS to see the effects.
"""
import copy
import csv
import itertools
import math
import random
from collections import defaultdict
from functools import partial
from itertools import groupby

from matplotlib import pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
# import matplotlib.cbook
import pandas as pd
import simpy
from simpy import *

# warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# General Settings
number = 2500  # Max number of jobs if infinite is false
noJobCap = True  # For infinite
maxTime = 10000.0  # Runtime limit
# Machine shop settings
processingTimes = [[6.75, 3.75, 2.5, 7.5], [3.75, 5.0, 7.5], [3.75, 2.5, 8.75, 5.0, 5.0]]
operationOrder = [[3, 1, 2, 5], [4, 1, 3], [2, 5, 1, 4, 3]]
numberOfOperations = [4, 3, 5]
machinesPerWC = [4, 2, 5, 3, 2]
machine_number_WC = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10, 11], [12, 13, 14], [15, 16]]
machine_number_relative = [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
                           [3, 0], [3, 1], [3, 2], [4, 0], [4, 1]]
setupTime = [[0, 0.625, 1.25], [0.625, 0, 0.8], [1.25, 0.8, 0]]
mean_setup = [0.515, 0.306, 0.515, 0.429, 0.306]
mean_processing_time = [5.875, 3.25, 6.6, 3.75, 4.5]
demand = [0.2, 0.5, 0.3]
noOfWC = range(len(machinesPerWC))

if noJobCap:
    number = 0

"Initial parameters of the GES"
noAttributes = 8
noAttributesJob = 4
totalAttributes = noAttributes + noAttributesJob

no_generation = 500


def list_duplicates(seq):
    tally = defaultdict(list)
    for ii, item in enumerate(seq):
        tally[item].append(ii)
    return ((key, locs) for key, locs in tally.items()
            if len(locs) >= 1)


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def bid_winner(env, jobs, noOfMachines, currentWC, job_shop, machine, store,
               normaliziation_range):
    """Used to calulcate the bidding values for each job in the pool, for each machine.
    Then checks which machine gets which job based on these bidding values."""
    current_bid = [0] * noOfMachines
    current_job = [0] * noOfMachines
    best_bid = []
    best_job = []
    no_of_jobs = len(jobs)
    total_rp = [0] * no_of_jobs

    # Get the remaning processing time
    for jj in range(no_of_jobs):
        total_rp[jj] = (remain_processing_time(jobs[jj]))

    # Get the bids for all machines
    for jj in range(noOfMachines):
        if job_shop.mach_av[(jj, currentWC - 1)] == 0:
            queue_length = len(machine[(jj, currentWC - 1)].items)
            new_bid = [0] * no_of_jobs
            for ii, job in enumerate(jobs):
                attributes = bid_calculation(job_shop.test_weights, machine_number_WC[currentWC - 1][jj],
                                             job.processingTime[job.currentOperation - 1], job.currentOperation,
                                             total_rp[ii], job.dueDate[job.numberOfOperations],
                                             env.now,
                                             job.priority, queue_length, normaliziation_range)

                new_bid[ii] = attributes

            ind_winning_job = new_bid.index(max(new_bid))
            current_bid[jj] = new_bid[ind_winning_job]
            current_job[jj] = ind_winning_job

        else:
            current_bid[jj] = -100_000
            current_job[jj] = 0

    if not all_equal(current_bid):

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

        # Put the job in the queue of the winning machine
        for ii, vv in enumerate(best_job):
            put_job_in_queue(currentWC, best_bid[ii], jobs[vv], job_shop, machine)

        # Remove job from queue of the JPA
        for ii in reversed(best_job):
            yield store.get(lambda mm: mm == jobs[ii])


def bid_calculation(weights_new, machinenumber, processing_time,
                    current, total_rp, due_date, now, job_priority, queue_length,
                    normalization):
    """Calulcates the bidding value of a job."""
    attribute = [0] * noAttributes
    attribute[0] = processing_time / 8.75 * weights_new[machinenumber - 1][0]  # processing time
    attribute[1] = (current - 1) / (5 - 1) * weights_new[machinenumber - 1][1]  # remaing operations
    attribute[2] = (due_date - now - normalization[0]) / (normalization[1] - normalization[0]) * \
                   weights_new[machinenumber - 1][2]  # RDue
    attribute[3] = total_rp / 21.25 * weights_new[machinenumber - 1][3]  # remaining processing time
    attribute[4] = (((due_date - now) / total_rp) - normalization[2]) / (normalization[3] - normalization[2]) * \
                   weights_new[machinenumber - 1][4]  # Critical Ratio
    attribute[5] = (job_priority - 1) / (10 - 1) * weights_new[machinenumber - 1][5]  # Job Priority
    attribute[6] = queue_length / 25 * weights_new[machinenumber - 1][6]  # Queue length
    attribute[7] = 0

    return sum(attribute)


def remain_processing_time(job):
    """Calculate the remaining processing time."""
    total_rp = 0
    for ii in range(job.currentOperation - 1, job.numberOfOperations):
        total_rp += job.processingTime[ii]

    return total_rp


def next_workstation(job, job_shop, env, min_job, max_job, max_wip):
    """Used to send a job to the next workstation or to complete the job.
    If a job has finished all of its operation, the relevant information (tardiness, flowtime)
    is stored. It is also checked if 2000 jobs have finished process, or if the max wip/time
    is exceded. In this, the end_event is triggered and the simulation is stopped."""
    if job.currentOperation + 1 <= job.numberOfOperations:
        job.currentOperation += 1
        nextWC = operationOrder[job.type - 1][job.currentOperation - 1]
        store = job_shop.storeWC[nextWC - 1]
        store.put(job)
    else:
        finish_time = env.now
        job_shop.totalWIP.append(job_shop.WIP)
        job_shop.tardiness[job.number] = max(job.priority * (finish_time - job.dueDate[job.numberOfOperations]), 0)
        job_shop.finishing_tard[job.number] = ([finish_time, job.dueDate[job.numberOfOperations], job.priority])

        job_shop.WIP -= 1
        job_shop.priority[job.number] = job.priority
        job_shop.flowtime[job.number] = finish_time - job.dueDate[0]
        # print(finish_time - job.dueDate[0])
        # finished_job += 1
        if job.number > max_job:
            if np.count_nonzero(job_shop.flowtime[min_job:max_job]) == 2000:
                job_shop.finish_time = env.now
                job_shop.end_event.succeed()

        if (job_shop.WIP > max_wip) | (env.now > 7_000):
            print("Fail", job_shop.WIP, env.now)
            job_shop.end_event.succeed()
            job_shop.early_termination = 1
            job_shop.finish_time = env.now


def set_makespan(current_makespan, job, env, setup_time, p):
    """Sets the makespan of a machine"""
    add = current_makespan + job.processingTime[job.currentOperation - 1] * p + setup_time

    new = env.now + job.processingTime[job.currentOperation - 1] * p + setup_time

    return max(add, new)


def put_job_in_queue(currentWC, choice, job, job_shop, machines):
    """Puts a job in a machine queue. Also checks if the machine is currently active
    or has a job in its queue. If not, it succeeds an event to tell the machine
    that a new job has been added to the queue."""
    machines[(choice, currentWC - 1)].put(job)
    if not job_shop.condition_flag[(choice, currentWC - 1)].triggered:
        job_shop.condition_flag[(choice, currentWC - 1)].succeed()


def choose_job_queue(weights_new_job, machinenumber, processing_time, due_date, env,
                     setup_time,
                     job_priority, normalization):
    """Calculates prioirities of jobs in a machines queue"""
    attribute_job = [0] * noAttributesJob

    attribute_job[3] = 0
    attribute_job[2] = setup_time / 1.25 * weights_new_job[machinenumber - 1][noAttributes + 2]
    attribute_job[1] = (job_priority - 1) / (10 - 1) * weights_new_job[machinenumber - 1][noAttributes + 1]
    attribute_job[0] = (due_date - processing_time - setup_time - env.now - normalization[4]) / (
            normalization[5] - normalization[4]) * \
                       weights_new_job[machinenumber - 1][noAttributes]

    return sum(attribute_job)


class Machine:
    def __init__(self, job_shop, current_WC, machine_number, env, weights_new, last_job, machine,
                 makespan, min_job, max_job, normalization, max_wip, time_to_failure, time_to_repair):
        self.env = env
        self.broken = False
        self.job_shop = job_shop

        self.process = env.process(
            self.machine_processing(job_shop, current_WC, machine_number, env, weights_new, last_job, machine,
                                    makespan, min_job, max_job, normalization, max_wip, time_to_repair))
        env.process(self.break_machine(time_to_failure, time_to_failure, job_shop, machine_number, current_WC))

    def machine_processing(self, job_shop, current_WC, machine_number, env, weights_new, last_job, machine,
                           makespan, min_job, max_job, normalization, max_wip, time_to_repair):
        """This refers to a Machine Agent in the system. It checks which jobs it wants to process
        next and stores relevant information regarding it."""
        ii = 0
        jj = 0
        relative_machine = machine_number_WC[current_WC - 1].index(machine_number)
        while True:
            if machine.items:
                setup_time = []
                priority_list = []
                if not last_job[relative_machine]:  # Only for the first job
                    ind_processing_job = 0
                    setup_time.append(0)
                else:
                    for job in machine.items:
                        setuptime = setupTime[job.type - 1][int(last_job[relative_machine]) - 1]
                        job_queue_priority = choose_job_queue(weights_new, machine_number,
                                                              job.processingTime[job.currentOperation - 1],
                                                              job.dueDate[job.currentOperation], env, setuptime,
                                                              job.priority,
                                                              normalization)  # Calulate the job priorities
                        priority_list.append(job_queue_priority)
                        setup_time.append(setuptime)
                    ind_processing_job = priority_list.index(max(priority_list))  # Get the job with the highest value

                next_job = machine.items[ind_processing_job]

                setuptime = setup_time[ind_processing_job]

                # proc_increase = next_job.proc_increase[
                #                          next_job.currentOperation - 1]
                proc_increase = 1

                time_in_processing = next_job.processingTime[
                                         next_job.currentOperation - 1] * proc_increase + setuptime  # Total time the machine needs to process the job

                job_shop.utilization[(relative_machine, current_WC - 1)] = job_shop.utilization[(
                    relative_machine, current_WC - 1)] + setuptime + next_job.processingTime[
                                                                               next_job.currentOperation - 1] * \
                                                                           proc_increase

                last_job[relative_machine] = next_job.type

                machine.items.remove(next_job)  # Remove job from queue

                job_shop.scheduleMachine[(relative_machine, current_WC - 1)].append([(next_job.processingTime[
                                                                                          next_job.currentOperation - 1] *
                                                                                      proc_increase),
                                                                                     setuptime, env.now,
                                                                                     env.now + time_in_processing])
                job_shop.job_per_machine[next_job.number][next_job.currentOperation - 1] = int(
                    machine_number_WC[current_WC - 1][relative_machine])
                job_shop.position_in_machine[next_job.number][next_job.currentOperation - 1] = jj
                jj += 1
                done_in = time_in_processing
                while done_in:
                    try:
                        start = env.now
                        yield env.timeout(done_in)
                        done_in = 0
                    except simpy.Interrupt:
                        self.broken = True
                        job_shop.mach_av[(relative_machine, current_WC - 1)] = 1
                        done_in -= self.env.now - start
                        yield self.env.timeout(time_to_repair[ii])
                        job_shop.mach_av[(relative_machine, current_WC - 1)] = 0
                        ii += 1
                        self.broken = False
                makespan[relative_machine] = env.now

                next_workstation(next_job, job_shop, env, min_job, max_job,
                                 max_wip)  # Send the job to the next workstation
            else:
                waiting = 1
                while waiting:
                    try:
                        yield job_shop.condition_flag[
                            (relative_machine,
                             current_WC - 1)]  # Used if there is currently no job in the machines queue
                        job_shop.condition_flag[(relative_machine, current_WC - 1)] = simpy.Event(
                            env)  # Reset event if it is used
                        waiting = 0
                    except simpy.Interrupt:
                        self.broken = True
                        # print("Int:" + str(env.now) + "WC:" + str(current_WC) + "Machine:" + str(relative_machine))
                        job_shop.mach_av[(relative_machine, current_WC - 1)] = 1
                        yield self.env.timeout(time_to_repair[ii - 1])
                        job_shop.mach_av[(relative_machine, current_WC - 1)] = 0
                        self.broken = False
                        ii += 1
                        waiting = 0

    def break_machine(self, time_to_failure, time_to_repair, job_shop, machine_number, current_WC):
        ii = 1
        # relative_machine = machine_number_WC[current_WC - 1].index(machine_number)
        while True:
            yield self.env.timeout(time_to_failure[ii] + time_to_repair[ii - 1])
            ii += 1
            if not self.broken:
                self.process.interrupt()


def cfp_wc(env, machine, store, job_shop, currentWC, normalization):
    """Sends out the Call-For-Proposals to the various machines.
    Represents the Job-Pool_agent"""
    while True:
        if store.items:
            # job_shop.QueuesWC[currentWC].append(
            #     {ii: len(job_shop.machine_per_wc[(ii, currentWC)].items) for ii in
            #      range(machinesPerWC[currentWC])})  # Stores the Queue length of the JPA
            c = bid_winner(env, store.items, machinesPerWC[currentWC], currentWC + 1, job_shop,
                           machine, store, normalization)
            env.process(c)
        tib = 0.5  # Frequency of when CFPs are sent out
        yield env.timeout(tib)


def source(env, number1, interval, job_shop, due_date_setting, min_job, rush_job_perc, proc_time_incr, proc_time_prob):
    """Reflects the Job Release Agent. Samples time and then "releases" a new
    job into the system."""
    while True:  # Needed for infinite case as True refers to "until".
        ii = number1
        number1 += 1
        job = New_Job('job%02d' % ii, env, ii, due_date_setting, job_shop, rush_job_perc, proc_time_incr,
                      proc_time_prob)
        if ii == min_job:
            job_shop.start_time = env.now  # Start counting when the minimum number of jobs have entered the system
        job_shop.tardiness.append(-1)
        job_shop.flowtime.append(0)
        job_shop.priority.append(0)
        job_shop.finishing_tard.append(0)
        job_shop.job_per_machine.append(np.zeros(job.numberOfOperations))
        job_shop.position_in_machine.append(np.zeros(job.numberOfOperations))
        job_shop.due_dates.append(job.dueDate)
        job_shop.priorities.append(job.priority)
        job_shop.proc_time_incr.append(job.proc_increase)
        job_shop.is_rush_job.append(job.due_date_reduction)
        job_shop.WIP += 1
        firstWC = operationOrder[job.type - 1][0]
        store = job_shop.storeWC[firstWC - 1]
        store.put(job)
        tib = random.expovariate(1.0 / interval)
        yield env.timeout(tib)


class jobShop:
    """This class creates a job shop, along with everything that is needed to run the Simpy Environment."""

    def __init__(self, env, weights_new):
        self.machine_per_wc = {(ii, jj): Store(env) for jj in noOfWC for ii in
                               range(machinesPerWC[jj])}  # Used to store jobs in a machine
        self.storeWC = {ii: FilterStore(env) for ii in noOfWC}  # Used to store jobs in a JPA
        self.QueuesWC = {jj: [] for jj in noOfWC}  # Can be used to keep track of Queue Lenghts
        self.scheduleWC = {ii: [] for ii in noOfWC}  # Used to keep track of the schedule
        self.makespanWC = {ii: np.zeros(machinesPerWC[ii]) for ii in
                           noOfWC}  # Keeps track of the makespan of each machine
        self.last_job_WC = {ii: np.zeros(machinesPerWC[ii]) for ii in
                            noOfWC}  # Keeps track of which job was last in the machine
        self.condition_flag = {(ii, jj): simpy.Event(env) for jj in noOfWC for ii in range(machinesPerWC[
                                                                                               jj])}  # An event which keeps track if a machine has had a job inserted into it if it previously had no job
        self.scheduleMachine = {(ii, jj): [] for jj in noOfWC for ii in
                                range(machinesPerWC[jj])}
        self.job_per_machine = []
        self.mach_av = {(ii, jj): 0 for jj in noOfWC for ii in
                        range(machinesPerWC[jj])}
        self.position_in_machine = []

        self.test_weights = weights_new
        self.flowtime = []  # Jobs flowtime
        self.tardiness = []  # Jobs tardiness
        self.WIP = 0  # Current WIP of the system
        self.early_termination = 0  # Whether the simulation terminated earlier
        self.utilization = {(ii, jj): 0 for jj in noOfWC for ii in range(machinesPerWC[jj])}
        self.finish_time = 0  # Finishing time of the system
        self.totalWIP = []  # Keeps track of the total WIP of the system

        self.bids = []  # Keeps track of the bids
        self.priority = []  # Keeps track of the job priorities
        self.start_time = 0  # Starting time of the simulation
        self.proc_time_incr = []
        # self.rush_job = []

        self.due_dates = []
        self.priorities = []
        self.finishing_tard = []
        self.is_rush_job = []


class New_Job:
    def __init__(self, name, env, number1, dueDateTightness, job_shop, rush_job_perce, proc_time_incre, proc_time_prob):
        jobType = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2], k=1)
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
        self.proc_increase = np.zeros(numberOfOperations[self.type - 1])
        p = random.choices([1, 0], weights=[rush_job_perce, 1 - rush_job_perce], k=1)

        # flowtime = np.nanmean([job_shop.flowtime[i] for i in range(self.number) if job_shop.flowtime[i] > 0])
        if all(job_shop.flowtime[i] == 0 for i in range(self.number)):
            flowtime = 0
        else:
            flowtime = np.mean([job_shop.flowtime[i] for i in range(self.number) if job_shop.flowtime[i] > 0])

        self.due_date_reduction = p[0] * flowtime * 0.3
        reduction_factor = (sum(processingTimes[self.type - 1]) * dueDateTightness - self.due_date_reduction) / (
                sum(processingTimes[self.type - 1]) * dueDateTightness)
        for ii in range(self.numberOfOperations):
            p_1 = random.choices([1 + proc_time_incre, 1], weights=[proc_time_prob, 1 - proc_time_prob], k=1)
            self.proc_increase[ii] = p_1[0]
            self.processingTime[ii] = processingTimes[self.type - 1][ii]
            self.dueDate[ii + 1] = self.dueDate[ii] + (self.processingTime[ii] * dueDateTightness) * reduction_factor

        # job_shop.proc_time_incr.append(self.proc_increase)


def get_breakdown_times(MTTA_factor, Ag):
    breakdowntimes = []
    repairtimes = []
    for wc in range(len(machinesPerWC)):
        for ii in range(machinesPerWC[wc]):
            machine_breakdown_times = [0]
            machine_repair_times = [0]
            while sum(machine_breakdown_times) < 20_000:
                MTTA_mean = MTTA_factor * mean_processing_time[wc]
                MTBA_mean = MTTA_mean / Ag - MTTA_mean

                MTTA = random.uniform(0, MTTA_mean)
                MTBA = random.expovariate(1 / MTBA_mean)

                machine_breakdown_times.append(MTBA)
                machine_repair_times.append(MTTA)
            breakdowntimes.append(machine_breakdown_times)
            repairtimes.append(machine_repair_times)

    return breakdowntimes, repairtimes


def get_measures(arrivalMean, due_date_tightness, job_shop, job_shop_AS, min_job, max_job1, new_tard):
    max_job = min(max_job1, np.count_nonzero(job_shop_AS.flowtime[min_job:max_job1]) + min_job)
    if max_job < max_job1:
        print(max_job)
    stability = np.zeros(4)
    if new_tard:
        objective1 = np.nanmean(new_tard[min_job:max_job]) + 0.01 * max(new_tard[min_job:max_job])
    else:
        objective1 = np.nanmean(job_shop_AS.tardiness[min_job:max_job]) + 0.01 * max(
            job_shop_AS.tardiness[min_job:max_job])
    objective = np.nanmean(job_shop.tardiness[min_job:max_job]) + 0.01 * max(job_shop.tardiness[min_job:max_job])

    # print(objective1, objective)

    if (arrivalMean == 1.5429) & (due_date_tightness == 6):
        robustness = objective1
    else:
        robustness = (np.abs(objective - objective1) / objective)

    if new_tard:
        for i in range(min_job, max_job):
            stability[0] += np.abs(new_tard[i] - job_shop.tardiness[i])
    else:
        for i in range(min_job, max_job):
            stability[0] += np.abs(job_shop_AS.tardiness[i] - job_shop.tardiness[i])

    for w in noOfWC:
        for m in range(machinesPerWC[w]):
            stability[1] += np.abs(
                np.sum(job_shop_AS.scheduleMachine[(m, w)], axis=0)[0] - np.sum(job_shop.scheduleMachine[(m, w)],
                                                                                axis=0))[0]
            stability[2] += np.abs(
                np.sum(job_shop_AS.scheduleMachine[(m, w)], axis=0)[1] - np.sum(job_shop.scheduleMachine[(m, w)],
                                                                                axis=0))[1]

    for ii in range(min_job, max_job):
        for jj in range(len(job_shop.job_per_machine[ii])):
            if job_shop.job_per_machine[ii][jj] == job_shop_AS.job_per_machine[ii][jj]:
                stability[3] += 0
            else:
                stability[3] += 1

    final_measures = [robustness]
    final_measures.extend(stability)

    return final_measures


def add_extra_time(new_schedule, current_job_ind, m, wc, repairtimes):
    for ii in range(current_job_ind, len(new_schedule[(m, wc)]) - 1):
        if repairtimes != 0:
            # repairtimes = max(repairtimes - (new_schedule[(m, wc)][ii + 1][2] - new_schedule[(m, wc)][ii][3]), 0)
            # print(repairtimes)
            if new_schedule[(m, wc)][ii + 1][2] - new_schedule[(m, wc)][ii][3] <= 0:
                # print(repairtimes)
                # new_schedule[(m, wc)][ii + 1][2] += repairtimes
                new_schedule[(m, wc)][ii + 1][2] = new_schedule[(m, wc)][ii][3]
                new_schedule[(m, wc)][ii + 1][3] = new_schedule[(m, wc)][ii + 1][2] + new_schedule[(m, wc)][ii + 1][0] + \
                                                   new_schedule[(m, wc)][ii + 1][1]
            # else:
            #     new_schedule[(m, wc)][ii + 1][2] += repairtimes
            #     repairtimes = max(repairtimes - (new_schedule[(m, wc)][ii + 1][2] - new_schedule[(m, wc)][ii][3]), 0)
            #     new_schedule[(m, wc)][ii][3] = new_schedule[(m, wc)][ii + 1][2]

    # print(new_schedule)
    return new_schedule


def right_shift_rush_jobs(job_shop, job_shop_AS, max_job):
    new_tard = []
    for ii in range(max_job):
        for jj in range(len(job_shop.job_per_machine[ii]) - 1):
            ft = job_shop.finishing_tard[ii][0]
            dd = job_shop.finishing_tard[ii][1]
            prio = job_shop.finishing_tard[ii][2]
            reduction = job_shop_AS.is_rush_job[ii]
            # print(reduction)
            new_tard.append(max(prio * (ft - (dd - reduction)), 0))
    # print(np.mean(new_tard))
    # print(np.mean(job_shop.tardiness))
    return new_tard


def right_shift_sched_proc_time(job_shop, breakdowntimes, repairtimes, max_job, new_schedule):
    new_schedule = copy.deepcopy(job_shop.scheduleMachine)
    for i in range(len(breakdowntimes)):
        breakdowntimes[i] = np.cumsum(breakdowntimes[i])
    rm = 0
    for ii in range(max_job):
        for jj in range(len(job_shop.job_per_machine[ii]) - 1):
            current_job_machine = job_shop.job_per_machine[ii][jj] - 1
            cj_wc = (machine_number_relative[int(current_job_machine)][0])
            cj_m = machine_number_relative[int(current_job_machine)][1]
            cj_ind = int(job_shop.position_in_machine[ii][jj])
            is_proc_incr_job = (job_shop.proc_time_incr[ii][jj])
            if is_proc_incr_job > 1:
                # print(is_proc_incr_job, cj_wc, cj_m, cj_ind)
                # print(new_schedule[(cj_m, cj_wc)][cj_ind][3])
                new_schedule[(cj_m, cj_wc)][cj_ind][3] = new_schedule[(cj_m, cj_wc)][cj_ind][2] + \
                                                         new_schedule[(cj_m, cj_wc)][cj_ind][1] + is_proc_incr_job * \
                                                         new_schedule[(cj_m, cj_wc)][cj_ind][0]
                # print(new_schedule[(cj_m, cj_wc)][cj_ind][3])
                new_schedule = add_extra_time(new_schedule, cj_ind, cj_m, cj_wc, 1)

    # print(new_schedule[(0, 0)])

    for ii in range(max_job):
        for jj in range(len(job_shop.job_per_machine[ii]) - 1):
            current_job_machine = job_shop.job_per_machine[ii][jj] - 1
            cj_wc = (machine_number_relative[int(current_job_machine)][0])
            cj_m = machine_number_relative[int(current_job_machine)][1]
            cj_ind = int(job_shop.position_in_machine[ii][jj])

            next_job_machine = job_shop.job_per_machine[ii][jj + 1] - 1
            nj_wc = machine_number_relative[int(next_job_machine)][0]
            nj_m = machine_number_relative[int(next_job_machine)][1]
            nj_ind = int(job_shop.position_in_machine[ii][jj + 1])

            if new_schedule[(cj_m, cj_wc)][cj_ind][3] > new_schedule[(nj_m, nj_wc)][nj_ind][2]:
                delta_time = new_schedule[(cj_m, cj_wc)][cj_ind][3] - new_schedule[(nj_m, nj_wc)][nj_ind][2]
                new_schedule[(nj_m, nj_wc)][nj_ind][2] += delta_time
                new_schedule[(nj_m, nj_wc)][nj_ind][3] = new_schedule[(nj_m, nj_wc)][nj_ind][2] + \
                                                         new_schedule[(nj_m, nj_wc)][nj_ind][0] + \
                                                         new_schedule[(nj_m, nj_wc)][nj_ind][1]
                new_schedule = add_extra_time(new_schedule, nj_ind + 1, nj_m, nj_wc, delta_time)

    tardiness = []
    for ii in range(len(job_shop.job_per_machine)):
        jj = len(job_shop.due_dates[ii]) - 2
        current_job_machine = job_shop.job_per_machine[ii][jj] - 1
        cj_wc = (machine_number_relative[int(current_job_machine)][0])
        cj_m = machine_number_relative[int(current_job_machine)][1]
        cj_ind = int(job_shop.position_in_machine[ii][jj])
        tardiness.append(max(
            job_shop.priority[ii] * (new_schedule[(cj_m, cj_wc)][cj_ind][3] - job_shop.due_dates[ii][jj + 1]),
            0))

    # print(tardiness)
    return tardiness


def right_shift_sched(job_shop, breakdowntimes, repairtimes, max_job, new_schedule):
    new_schedule = copy.deepcopy(job_shop.scheduleMachine)
    for i in range(len(breakdowntimes)):
        breakdowntimes[i] = np.cumsum(breakdowntimes[i])
    rm = 0
    for wc in range(len(machinesPerWC)):
        for m in range(machinesPerWC[wc]):
            jj = 1
            for ii in range(len(new_schedule[(m, wc)]) - 1):
                if (new_schedule[(m, wc)][ii][2] < breakdowntimes[rm][jj]) & (
                        new_schedule[(m, wc)][ii][3] > breakdowntimes[rm][jj]):
                    new_schedule[(m, wc)][ii][3] += repairtimes[rm][jj]
                    new_schedule = add_extra_time(new_schedule, ii, m, wc, repairtimes[rm][jj])
                    jj += 1
                elif (new_schedule[(m, wc)][ii][3] < breakdowntimes[rm][jj]) & (
                        new_schedule[(m, wc)][ii + 1][2] > breakdowntimes[rm][jj]):
                    new_schedule[(m, wc)][ii + 1][2] += repairtimes[rm][jj] - (
                            new_schedule[(m, wc)][ii + 1][2] - breakdowntimes[rm][jj])
                    new_schedule[(m, wc)][ii + 1][3] = new_schedule[(m, wc)][ii + 1][2] + new_schedule[(m, wc)][ii + 1][
                        0] + new_schedule[(m, wc)][ii + 1][1]
                    new_schedule = add_extra_time(new_schedule, ii + 1, m, wc, repairtimes[rm][jj])
                    jj += 1
                while breakdowntimes[rm][jj] < new_schedule[(m, wc)][ii + 1][2]:
                    jj += 1
            rm += 1
            # jj = 1

    for ii in range(max_job):
        for jj in range(len(job_shop.job_per_machine[ii]) - 1):
            current_job_machine = job_shop.job_per_machine[ii][jj] - 1
            cj_wc = (machine_number_relative[int(current_job_machine)][0])
            cj_m = machine_number_relative[int(current_job_machine)][1]
            cj_ind = int(job_shop.position_in_machine[ii][jj])

            next_job_machine = job_shop.job_per_machine[ii][jj + 1] - 1
            nj_wc = machine_number_relative[int(next_job_machine)][0]
            nj_m = machine_number_relative[int(next_job_machine)][1]
            nj_ind = int(job_shop.position_in_machine[ii][jj + 1])

            if new_schedule[(cj_m, cj_wc)][cj_ind][3] > new_schedule[(nj_m, nj_wc)][nj_ind][2]:
                delta_time = new_schedule[(cj_m, cj_wc)][cj_ind][3] - new_schedule[(nj_m, nj_wc)][nj_ind][2]
                new_schedule[(nj_m, nj_wc)][nj_ind][2] += delta_time
                new_schedule[(nj_m, nj_wc)][nj_ind][3] = new_schedule[(nj_m, nj_wc)][nj_ind][2] + \
                                                         new_schedule[(nj_m, nj_wc)][nj_ind][0] + \
                                                         new_schedule[(nj_m, nj_wc)][nj_ind][1]
                new_schedule = add_extra_time(new_schedule, nj_ind + 1, nj_m, nj_wc, delta_time)

    tardiness = []
    for ii in range(len(job_shop.job_per_machine)):
        jj = len(job_shop.due_dates[ii]) - 2
        current_job_machine = job_shop.job_per_machine[ii][jj] - 1
        cj_wc = (machine_number_relative[int(current_job_machine)][0])
        cj_m = machine_number_relative[int(current_job_machine)][1]
        cj_ind = int(job_shop.position_in_machine[ii][jj])
        tardiness.append(max(
            job_shop.priority[ii] * (new_schedule[(cj_m, cj_wc)][cj_ind][3] - job_shop.due_dates[ii][jj + 1]),
            0))

    # print(tardiness)
    return tardiness


def do_simulation_with_weights(mean_weight_new, arrivalMean, due_date_tightness, normalization,
                               min_job, max_job, max_wip, rush_job_percentage, proc_time_increase, proc_time_prob,
                               repair_time_ind, breakdown_prob,
                               iter):
    random.seed(iter)
    breakdowntimes, repairtimes = get_breakdown_times(repair_time_ind, breakdown_prob)
    # print(len(breakdowntimes[0]))
    breakdowntimes1 = [[0, 30_000]] * 16
    random.seed(iter)
    env = Environment()  # Create Environment
    job_shop = jobShop(env, mean_weight_new)  # Initiate the job shop
    env.process(source(env, 0, arrivalMean, job_shop, due_date_tightness,
                       min_job, 0, proc_time_increase, proc_time_prob))  # Starts the source (Job Release Agent)

    for wc in range(len(machinesPerWC)):
        last_job = job_shop.last_job_WC[wc]
        makespanWC = job_shop.makespanWC[wc]
        store = job_shop.storeWC[wc]

        env.process(
            cfp_wc(env, job_shop.machine_per_wc, store, job_shop, wc, normalization))

        for ii in range(machinesPerWC[wc]):
            machine = job_shop.machine_per_wc[(ii, wc)]
            machines = Machine(job_shop, wc + 1, machine_number_WC[wc][ii], env, mean_weight_new, last_job,
                               machine, makespanWC, min_job, max_job, normalization, max_wip,
                               breakdowntimes1[machine_number_WC[wc][ii] - 1],
                               repairtimes[machine_number_WC[wc][ii] - 1])

            # env.process(
            #     machine_processing(job_shop, wc + 1, machine_number_WC[wc][ii], env, mean_weight_new, last_job,
            #                        machine, makespanWC, min_job, max_job, normalization, max_wip))
    job_shop.end_event = env.event()

    env.run(until=job_shop.end_event)  # Run the simulation until the end event gets triggered

    # new_tard = []

    # new_tard = right_shift_sched_proc_time(job_shop, breakdowntimes, repairtimes, max_job, job_shop.scheduleMachine)
    # new_tard = right_shift_rush_jobs(job_shop, max_job)

    random.seed(iter)
    env = Environment()  # Create Environment
    job_shop_AS = jobShop(env, mean_weight_new)  # Initiate the job shop
    env.process(source(env, 0, arrivalMean, job_shop_AS, due_date_tightness,
                       min_job, rush_job_percentage, proc_time_increase, proc_time_prob, ))  # Starts the source (Job Release Agent)

    # machines = np.zeros(())

    for wc in range(len(machinesPerWC)):
        last_job = job_shop_AS.last_job_WC[wc]
        makespanWC = job_shop_AS.makespanWC[wc]
        store = job_shop_AS.storeWC[wc]

        env.process(
            cfp_wc(env, job_shop_AS.machine_per_wc, store, job_shop_AS, wc, normalization))

        for ii in range(machinesPerWC[wc]):
            machine = job_shop_AS.machine_per_wc[(ii, wc)]
            machines = Machine(job_shop_AS, wc + 1, machine_number_WC[wc][ii], env, mean_weight_new, last_job,
                               machine, makespanWC, min_job, max_job, normalization, max_wip + 150,
                               breakdowntimes1[machine_number_WC[wc][ii] - 1],
                               repairtimes[machine_number_WC[wc][ii] - 1])
            #
            # env.process(
            #     machine_processing(job_shop_AS, wc + 1, machine_number_WC[wc][ii], env, mean_weight_new, last_job,
            #                        machine, makespanWC, min_job, max_job, normalization, max_wip))
    job_shop_AS.end_event = env.event()

    env.run(until=job_shop_AS.end_event)  # Run the simulation until the end event gets triggered
    new_tard = []
    # new_tard = right_shift_rush_jobs(job_shop, job_shop_AS, max_job)
    # # #
    return get_measures(arrivalMean, due_date_tightness, job_shop, job_shop_AS, min_job, max_job, new_tard)
    # return np.mean(job_shop.tardiness[min_job:max_job]) - np.mean(new_tard[min_job:max_job])


if __name__ == '__main__':

    arrival_time = [1.5429, 1.5429, 1.4572, 1.4572, 1.3804, 1.3804]
    utilization = [85, 85, 90, 90, 95, 95]
    due_date_settings = [4, 6, 4, 6, 4, 6]

    min_jobs = [499, 499, 999, 999, 1499, 1499]
    max_jobs = [2499, 2499, 2999, 2999, 3499, 3499]
    wip_max = [150, 150, 200, 200, 300, 300]

    normaliziation = [[-75, 150, -8, 12, -75, 150], [-30, 150, -3, 12, -30, 150], [-200, 150, -15, 12, -200, 150],
                      [-75, 150, -8, 12, -75, 150], [-300, 150, -50, 12, -300, 150], [-150, 150, -15, 12, -150, 150]]

    rush_job_perc = [0.00, 0.05, 0.10, 0.15, 0.20]

    proc_time_incr = [0, 0.05, 0.10]
    proc_incr_prob = [0, 0.05, 0.10, 0.15, 0.20]

    repair_time = [1, 5, 10]
    unav_prob = [0.05, 0.10, 0.15, 0.20]

    final_obj = []
    final_std = []

    no_runs = 50
    no_processes = 5

    for j in range(len(utilization)):
        results = []
        for d in range(1, len(rush_job_perc)):
        # for d, e in itertools.product(range(1, len(proc_time_incr)), range(1, len(proc_incr_prob))):
        #     print(proc_time_incr[d], proc_incr_prob[e])
            print(rush_job_perc[d])
            final_result = []
            str1 = "Runs/Final_runs/Run-weights-" + str(utilization[j]) + "-" + str(due_date_settings[j]) + ".csv"
            df = pd.read_csv(str1, header=None)
            weights = df.values.tolist()
            obj = np.zeros(no_runs)
            for k in range(int(no_runs / no_processes)):
                jobshop_pool = Pool(processes=no_processes)
                seeds = range(k * no_processes, k * no_processes + no_processes)
                func1 = partial(do_simulation_with_weights, weights, arrival_time[j], due_date_settings[j],
                                normaliziation[j], min_jobs[j], max_jobs[j], wip_max[j], rush_job_perc[d],
                                proc_time_incr[0], proc_incr_prob[0], repair_time[0], unav_prob[0])
                makespan_per_seed = jobshop_pool.map(func1, seeds)
                # print(makespan_per_seed)
                for h in range(no_processes):
                    final_result.append(makespan_per_seed[h])
            results.append(np.mean(final_result, axis=0))
            print(results)
        #
        print(results)
        results1 = pd.DataFrame(results,
                                columns=['Robustness', 'Stability 1', 'Stability 2', 'Stability 3', 'Stability 4'])
        file_name = f"Results/Rush_Job{utilization[j]}_{due_date_settings[j]}.csv"
        results1.to_csv(file_name)

        # arrival_time = [1.5429, 1.5429, 1.5429, 1.4572, 1.4572, 1.4572, 1.3804, 1.3804, 1.3804]

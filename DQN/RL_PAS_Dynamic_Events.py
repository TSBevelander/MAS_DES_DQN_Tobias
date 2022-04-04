"""
DES of the MAS. This DES is used to optimize certain aspects of the
MAS such as the bids. It can be used to quickly run multiple experiments.
The results can then be implemented in the MAS to see the effects.
"""
import itertools
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import simpy
from simpy import *
from statistics import mean
from joblib import load

# warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# General Settings
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

"Initial parameters of the GES"
noAttributes = 8
noAttributesJob = 4
totalAttributes = noAttributes + noAttributesJob

FTmodel = load('FTmodel85.joblib')
TDmodel = load('TDmodel85.joblib')

class jobShop:
    """This class creates a job shop, along with everything that is needed to run the Simpy Environment."""

    def list_duplicates(self,seq,BidMach):
        tally = defaultdict(list)
        for ii, item in enumerate(seq):
            tally[item].append(BidMach[ii])
        return ((key, locs) for key, locs in tally.items()
                if len(locs) >= 1)

    def tardiness_estimation(self, new_job, machine_queue, machine_number, weights, makespan, last_job, normalization):
        makespan = max(makespan,self.env.now)
        joblist = [[new_job.number, new_job.type, new_job.priority, new_job.processingTime[0], new_job.dueDate[1], 0]]
        jobs = [0]
        endtimes = [0]
        jobsi = 0
        for job in machine_queue:
            jobsi += 1
            joblist.append([job.number, job.type, job.priority, job.processingTime[0], job.dueDate[1], jobsi])
            jobs.append(jobsi)
            endtimes.append(0)
        if len(machine_queue) > 0:
                while jobs[0] == 0:
                    new_joblist = [joblist[x] for x in jobs]
                    setup_time = []
                    priority_list = []
                    for job in new_joblist:
                        setuptime = setupTime[job[1]-1][last_job-1]
                        job_queue_priority = self.choose_job_queue(weights, machine_number, job[3], job[4],setuptime, job[2], makespan, normalization)
                        priority_list.append(job_queue_priority)
                        setup_time.append(setuptime)
                    ind_processing_job = priority_list.index(max(priority_list))
                    makespan = makespan + setup_time[ind_processing_job] + new_joblist[ind_processing_job][3]
                    last_job = new_joblist[ind_processing_job][1]
                    endtimes[jobs[ind_processing_job]] = makespan
                    del jobs[ind_processing_job]
                    if len(jobs) == 0:
                        break
        else:
            setuptime = setupTime[new_job.type-1][last_job-1]
            endtimes[0] = makespan + setuptime + new_job.processingTime[0]
        tardiness = max(new_job.priority * (endtimes[0] - new_job.dueDate[1]), 0)
        return tardiness


    def choose_job_queue(self, weights_new_job, machine_number, processing_time, due_date,
                                 setup_time,
                                 job_priority, makespan, normalization):
        """Calculates prioirities of jobs in a machines queue"""
        attribute_job = [0] * noAttributesJob

        attribute_job[3] = 0
        attribute_job[2] = setup_time / 1.25 * weights_new_job[machine_number][noAttributes + 2]
        attribute_job[1] = (job_priority - 1) / (10 - 1) * weights_new_job[machine_number][noAttributes + 1]
        attribute_job[0] = (due_date - processing_time - setup_time - makespan - normalization[4]) / (
                normalization[5] - normalization[4]) * \
                           weights_new_job[machine_number][noAttributes]
        return sum(attribute_job)

    def decision_tree_input(self,job,machine,queue):
        t0p0 = 0
        t0p1 = 0
        t0p2 = 0
        t1p0 = 0
        t1p1 = 0
        t1p2 = 0
        t2p0 = 0
        t2p1 = 0
        t2p2 = 0
        jobsN = 0
        for jobs in queue:
            if jobs.type == 1:
                if jobs.priority == 1:
                    t0p0 += 1
                elif jobs.priority == 3:
                    t0p1 += 1
                elif jobs.priority == 10:
                    t0p2 += 1
            elif jobs.type == 2:
                if jobs.priority == 1:
                    t1p0 += 1
                elif jobs.priority == 3:
                    t1p1 += 1
                elif jobs.priority == 10:
                    t1p2 += 1
            elif jobs.type == 3:
                if jobs.priority == 1:
                    t2p0 += 1
                elif jobs.priority == 3:
                    t2p1 += 1
                elif jobs.priority == 10:
                    t2p2 += 1
            jobsN += 1
        if job.priority == 1:
            job_priority = 0
        elif job.priority == 3:
            job_priority = 1
        elif job.priority == 10:
            job_priority = 2
        result = [machine, job.type, job_priority, jobsN, t0p0, t0p1, t0p2, t1p0, t1p1, t1p2, t2p0, t2p1, t2p2]
        return result

    def tardiness_estimation2(self, prediction_data, job):
        FT = FTmodel.predict([prediction_data])
        finish_time = self.env.now + FT[0]
        job_tardiness = max(job.priority * (finish_time - job.dueDate[1]), 0)
        return job_tardiness

    def bid_winner(self, jobs, noOfMachines, BiddingMachines, currentWC, machine, store,
                   normaliziation_range, weights):
        """Used to calulcate the bidding values for each job in the pool, for each machine.
        Then checks which machine gets which job based on these bidding values."""
        current_bid = [0] * 3
        current_job = [0] * noOfMachines
        best_bid = []
        best_job = []
        no_of_jobs = len(jobs)
        total_rp = [0] * no_of_jobs


        # Get the remaning processing time
        for jj in range(no_of_jobs):
            total_rp[jj] = (self.remain_processing_time(jobs[jj]))


        # Get the bids for all machines
        for jj, mm in enumerate(BiddingMachines):
            queue_length = len(machine[(mm, currentWC)].items)
            new_bid = [0] * no_of_jobs
            for ii, job in enumerate(jobs):
                attributes = self.bid_calculation(self.test_weights, machine_number_WC[currentWC][mm],
                                                  job.processingTime[job.currentOperation - 1], job.currentOperation,
                                                  total_rp[ii], job.dueDate[job.numberOfOperations],
                                                  self.env.now,
                                                  job.priority, queue_length, normaliziation_range)

                new_bid[ii] = attributes

            ind_winning_job = new_bid.index(max(new_bid))
            current_bid[jj] = new_bid[ind_winning_job]
            current_job[jj] = ind_winning_job

        # Determine the winning bids
        sorted_list = sorted(self.list_duplicates(current_job,BiddingMachines))
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
        self.cfp_tardiness = 0
        for ii, vv in enumerate(best_job):
            rtuf = self.remaining_time_until_free(machine[(best_bid[ii],0)],self.makespanWC[0][best_bid[ii]],self.last_job_WC[0][best_bid[ii]])
            if machine[(best_bid[ii], 0)].items:
                if machine[(best_bid[ii], 0)].items[len(machine[(best_bid[ii], 0)].items)-1].type is not jobs[vv].type:
                    rtuf = rtuf + setupTime[jobs[vv].type-1][machine[(best_bid[ii], 0)].items[len(machine[(best_bid[ii], 0)].items)-1].type - 1]
            else:
                if int(self.last_job_WC[0][best_bid[ii]]) is not jobs[vv].type:
                    rtuf = rtuf + setupTime[jobs[vv].type-1][int(self.last_job_WC[0][best_bid[ii]]-1)]
            job_end_time = self.env.now + rtuf + jobs[vv].processingTime[0]
            job_tardiness = max(jobs[vv].priority * (job_end_time - jobs[vv].dueDate[1]), 0)
            self.rewards[jobs[vv].number][0] = self.tardiness_estimation(jobs[vv], machine[(best_bid[ii],0)].items, best_bid[ii], weights, self.machines[best_bid[ii]].makespan, self.machines[best_bid[ii]].last_job, self.normalization)
            self.cfp_tardiness = self.cfp_tardiness + job_tardiness
            if jobs[vv].number >= self.min_job and jobs[vv].number < self.max_job:
                self.flow_time_estimate_input[jobs[vv].number-self.min_job] = self.decision_tree_input(jobs[vv],best_bid[ii], machine[(best_bid[ii],0)].items)
            self.rewards[jobs[vv].number][1] = self.tardiness_estimation2(self.decision_tree_input(jobs[vv],best_bid[ii], machine[(best_bid[ii],0)].items),jobs[vv])
            self.rewards[jobs[vv].number][2] = max(TDmodel.predict([self.decision_tree_input(jobs[vv],best_bid[ii], machine[(best_bid[ii],0)].items)])[0],0)
            self.put_job_in_queue(currentWC+1, best_bid[ii], jobs[vv], machine)
            self.cfptime[jobs[vv].number] = self.env.now
        #self.cfp_tardiness = self.cfp_tardiness/len(best_job)

        # Remove job from queue of the JPA
        for ii in reversed(best_job):
            yield store.get(lambda mm: mm == jobs[ii])



    def bid_calculation(self, weights_new, machinenumber, processing_time,
                        current, total_rp, due_date, now, job_priority, queue_length,
                        normalization):
        """Calulcates the bidding value of a job."""
        attribute = [0] * noAttributes
        attribute[0] = processing_time / 7.5 * weights_new[machinenumber - 1][0]  # processing time
        attribute[1] = (current - 1) / (5 - 1) * weights_new[machinenumber - 1][1]  # remaing operations
        attribute[2] = (due_date - now - normalization[0]) / (normalization[1] - normalization[0]) * \
                       weights_new[machinenumber - 1][2]  # slack
        attribute[3] = total_rp / 7.5 * weights_new[machinenumber - 1][3]  # remaining processing time
        attribute[4] = (((due_date - now) / total_rp) - normalization[2]) / (normalization[3] - normalization[2]) * \
                       weights_new[machinenumber - 1][4]  # Critical Ratio
        attribute[5] = (job_priority - 1) / (10 - 1) * weights_new[machinenumber - 1][5]  # Job Priority
        attribute[6] = queue_length / 30 * weights_new[machinenumber - 1][6]  # Queue length
        attribute[7] = 0

        return sum(attribute)


    def remain_processing_time(self, job):
        """Calculate the remaining processing time."""
        total_rp = 0
        for ii in range(job.currentOperation - 1, job.numberOfOperations):
            total_rp += job.processingTime[ii]

        return total_rp


    def next_workstation(self, job,min_job, max_job, max_wip, machine_number):
        """Used to send a job to the next workstation or to complete the job.
        If a job has finished all of its operation, the relevant information (tardiness, flowtime)
        is stored. It is also checked if 2000 jobs have finished process, or if the max wip/time
        is exceded. In this, the end_event is triggered and the simulation is stopped."""
        if job.currentOperation + 1 <= job.numberOfOperations:
            job.currentOperation += 1
            nextWC = operationOrder[job.type - 1][job.currentOperation - 1]
            store = self.storeWC[nextWC - 1]
            store.put(job)
        else:
            finish_time = self.env.now
            self.totalWIP.append(self.WIP)
            self.tardiness[job.number] = max(job.priority * (finish_time - job.dueDate[job.numberOfOperations]), 0)

            self.WIP -= 1
            self.machinelog[job.number] = machine_number
            self.priority[job.number] = job.priority
            self.flowtime[job.number] = finish_time - job.dueDate[0]
            if job.priority == 1:
                job_priority = 0
            elif job.priority == 3:
                job_priority = 1
            elif job.priority == 10:
                job_priority = 2
            self.running_flowtime[machine_number-1][job.type-1][job_priority] = [(self.running_flowtime[machine_number-1][job.type-1][job_priority][0]*self.running_flowtime[machine_number-1][job.type-1][job_priority][1]+self.flowtime[job.number])/(self.running_flowtime[machine_number-1][job.type-1][job_priority][1]+1),self.running_flowtime[machine_number-1][job.type-1][job_priority][1]+1]
            if job.number >= min_job and job.number < max_job:
                self.flow_time_estimate_output[job.number-min_job] = self.flowtime[job.number]
                self.tardiness_estimate_output[job.number-min_job] = job.priority * (finish_time - job.dueDate[job.numberOfOperations])
            # finished_job += 1
            if job.number > max_job:
                if np.count_nonzero(self.flowtime[min_job:max_job]) == 2000:
                    self.finish_time = self.env.now
                    if not self.end_event.triggered:
                        self.end_event.succeed()

            if (self.WIP > max_wip) | (self.env.now > 30_000):
                if not self.end_event.triggered:
                    self.end_event.succeed()
                self.early_termination = 1
                self.finish_time = self.env.now

    def set_makespan(self, current_makespan, job, setup_time):
        """Sets the makespan of a machine"""
        add = current_makespan + job.processingTime[job.currentOperation - 1] + setup_time

        new = self.env.now + job.processingTime[job.currentOperation - 1] + setup_time

        return max(add, new)


    def put_job_in_queue(self, currentWC, choice, job, machines):
        """Puts a job in a machine queue. Also checks if the machine is currently active
        or has a job in its queue. If not, it succeeds an event to tell the machine
        that a new job has been added to the queue."""
        machines[(choice, currentWC - 1)].put(job)
        if not self.condition_flag[(choice, currentWC - 1)].triggered:
            self.condition_flag[(choice, currentWC - 1)].succeed()

    class Machine:
        def __init__(self,job_shop, current_WC, machine_number, weights_new, last_job, machine,
                               makespan, min_job, max_job, normalization, max_wip, time_to_failure, time_to_repair):
            self.env = job_shop.env
            self.broken = False
            self.makespan = int(makespan[machine_number-1])
            self.last_job = int(last_job[machine_number-1])
            self.process = job_shop.env.process(self.machine_processing(job_shop, current_WC, machine_number, weights_new, self.last_job, machine,
                               self.makespan, min_job, max_job, normalization, max_wip, time_to_repair))

            job_shop.env.process(self.break_machine(job_shop, time_to_failure,time_to_repair,machine_number))

        def choose_job_queue_machine(self, weights_new_job, machine_number, processing_time, due_date,
                             setup_time,
                             job_priority, normalization):
            """Calculates prioirities of jobs in a machines queue"""
            attribute_job = [0] * noAttributesJob

            attribute_job[3] = 0
            attribute_job[2] = setup_time / 1.25 * weights_new_job[machine_number - 1][noAttributes + 2]
            attribute_job[1] = (job_priority - 1) / (10 - 1) * weights_new_job[machine_number - 1][noAttributes + 1]
            attribute_job[0] = (due_date - processing_time - setup_time - self.env.now - normalization[4]) / (
                    normalization[5] - normalization[4]) * \
                               weights_new_job[machine_number - 1][noAttributes]

            return sum(attribute_job)


        def machine_processing(self,job_shop, current_WC, machine_number, weights_new, last_job, machine,
                               makespan,min_job, max_job, normalization, max_wip, time_to_repair):
            """This refers to a Machine Agent in the system. It checks which jobs it wants to process
            next and stores relevant information regarding it."""
            ii = 1
            while True:
                relative_machine = machine_number_WC[current_WC - 1].index(machine_number)
                if machine.items:
                    setup_time = []
                    priority_list = []
                    if not last_job:  # Only for the first job
                        ind_processing_job = 0
                        setup_time.append(0)
                    else:
                        for job in machine.items:
                            setuptime = setupTime[job.type - 1][int(last_job) - 1]
                            job_queue_priority = self.choose_job_queue_machine(weights_new, machine_number,
                                                                       job.processingTime[job.currentOperation - 1],
                                                                       job.dueDate[job.currentOperation], setuptime,
                                                                       job.priority, normalization)  # Calulate the job priorities
                            priority_list.append(job_queue_priority)
                            setup_time.append(setuptime)
                        ind_processing_job = priority_list.index(max(priority_list))

                    next_job = machine.items[ind_processing_job]
                    setuptime = setup_time[ind_processing_job]
                    time_in_processing = next_job.processingTime[
                                             next_job.currentOperation - 1] + setuptime  # Total time the machine needs to process the job

                    makespan = job_shop.set_makespan(makespan, next_job, setuptime)

                    job_shop.utilization[(relative_machine, current_WC - 1)] = job_shop.utilization[(
                        relative_machine, current_WC - 1)] + setuptime + next_job.processingTime[next_job.currentOperation - 1]
                    last_job = next_job.type
                    machine.items.remove(next_job)  # Remove job from queue

                    done_in = time_in_processing
                    while done_in:
                        try:
                            start = job_shop.env.now
                            yield job_shop.env.timeout(done_in)
                            done_in = 0
                        except simpy.Interrupt:
                            self.broken = True
                            done_in -= job_shop.env.now - start
                            yield job_shop.env.timeout(time_to_repair[machine_number-1][ii])
                            ii += 1
                            job_shop.last_repair[machine_number-1] = job_shop.env.now
                            self.broken = False
                    makespan = job_shop.env.now

                    job_shop.next_workstation(next_job, min_job, max_job, max_wip, machine_number)  # Send the job to the next workstation
                else:
                    waiting = 1
                    while waiting:
                        try:
                            yield job_shop.condition_flag[
                        (relative_machine, current_WC - 1)]  # Used if there is currently no job in the machines queue
                            job_shop.condition_flag[(relative_machine, current_WC - 1)] = simpy.Event(job_shop.env)  # Reset event if it is used
                            waiting = 0
                        except simpy.Interrupt:
                            self.broken = True
                            yield job_shop.env.timeout(time_to_repair[machine_number-1][ii])
                            job_shop.last_repair[machine_number - 1] = job_shop.env.now
                            self.broken = False
                            ii += 1

        def break_machine(self, job_shop, time_to_failure, time_to_repair, machine_number):
            ii = 1
            while True:
                yield job_shop.env.timeout(time_to_failure[machine_number-1][ii] + time_to_repair[machine_number-1][ii])
                ii += 1
                if not self.broken:
                    self.process.interrupt()

    def source(self, number1, interval, due_date_setting, min_job):
        """Reflects the Job Release Agent. Samples time and then "releases" a new
        job into the system."""
        while True:  # Needed for infinite case as True refers to "until".
            ii = number1
            number1 += 1
            job = New_Job('job%02d' % ii, self.env, ii, due_date_setting)
            if ii == min_job:
                self.env.start_time = self.env.now
            self.tardiness.append(-1)
            self.machinelog.append(-1)
            if len(self.interarrivaltimes)==0:
                self.interarrivaltimes.append(self.env.now)
            else:
                self.interarrivaltimes.append(self.env.now-self.arrivaltimes[-1])
            self.arrivaltimes.append(self.env.now)
            self.cfptime.append(-1)
            self.rewards.append([-1,-1,-1])
            self.flowtime.append(0)
            self.priority.append(0)
            if ii >= self.min_job and ii < self.max_job:
                self.flow_time_estimate_output.append(0)
                self.flow_time_estimate_input.append(0)
                self.tardiness_estimate_output.append(0)
            self.WIP += 1
            firstWC = operationOrder[job.type - 1][0]
            store = self.storeWC[firstWC - 1]
            store.put(job)
            tib = random.expovariate(1.0 / interval)
            yield self.env.timeout(tib)

    def cfp_wc_trigger(self, store):
        while True:
            if store.items:
                self.items_to_bid.succeed()
                self.items_to_bid = self.env.event()
            tib = 0.4
            yield self.env.timeout(tib)

    def action_translate(self,action):
        if action == 0:
            NumBiddingMachines = 1
            BiddingMachines = [0]
        elif action == 1:
            NumBiddingMachines = 1
            BiddingMachines = [1]
        elif action == 2:
            NumBiddingMachines = 1
            BiddingMachines = [2]
        elif action == 3:
            NumBiddingMachines = 2
            BiddingMachines = [0, 1]
        elif action == 4:
            NumBiddingMachines = 2
            BiddingMachines = [0, 2]
        elif action == 5:
            NumBiddingMachines = 2
            BiddingMachines = [1, 2]
        elif action == 6:
            NumBiddingMachines = 3
            BiddingMachines = [0, 1, 2]
        return NumBiddingMachines, BiddingMachines

    def step(self,action):
        NumBiddingMachines, BiddingMachines = self.action_translate(action)
        c = self.bid_winner(self.storeWC[0].items[:10], NumBiddingMachines, BiddingMachines, 0,self.machine_per_wc, self.storeWC[0], self.normalization, self.test_weights)
        self.env.process(c)


        terminal = True if self.end_event.triggered else False
        reward = self.cfp_tardiness

        self.env.run(until=self.items_to_bid)

        observation = self.get_environment_state(self.storeWC[0],self.machine_per_wc)
        return observation, terminal, reward

    def get_environment_state(self, store, machines):
        observation = []
        item_amount = min(len(store.items),10)
        observation.append(item_amount)
        for ii in range(item_amount):
            observation.append(store.items[ii].type)
            observation.append(store.items[ii].priority)
            observation.append(store.items[ii].processingTime[0])
            observation.append(max(store.items[ii].dueDate[1]-self.env.now,0))
        for ii in range(item_amount,10):
            observation.append(0)
            observation.append(0)
            observation.append(10)
            observation.append(100)
        for ii in range(3):
            machine = machines[(ii,0)]
            observation.append(len(machine.items))
            rtuf = self.remaining_time_until_free(machine,self.makespanWC[0][ii],self.last_job_WC[0][ii])
            observation.append(rtuf)
            observation.append(self.last_job_WC[0][ii])
            observation.append(self.env.now-self.last_repair[ii])
            if self.machines[ii].broken == True:
                observation.append(1)
            else:
                observation.append(0)
        if len(self.interarrivaltimes)==0:
            RIAT = 0
        else:
            RIAT = np.mean(self.interarrivaltimes[max(len(self.interarrivaltimes)-50,0):len(self.interarrivaltimes)])
        observation.append(RIAT)
        observation.append(self.WIP)
        observation.append(self.env.now)
        observation.append(0)
        return observation

    def remaining_time_until_free(self,machine,makespan,last_job):
        if len(machine.items) != 0:
            for ii in range(len(machine.items)):
                if (last_job != machine.items[ii].type) and (last_job != 0):
                    makespan = makespan + setupTime[machine.items[ii].type-1][int(last_job) - 1]
                last_job = machine.items[ii].type
                makespan = makespan + machine.items[ii].processingTime[0]
        rtuf = max(makespan - self.env.now,0)
        return rtuf

    def get_breakdown_times(self,MTTA_factor,Ag):
        breakdowntimes = []
        repairtimes = []
        for wc in range(len(machinesPerWC)):
            for ii in range(machinesPerWC[wc]):
                machine_breakdown_times = []
                machine_repair_times = []
                while sum(machine_breakdown_times) < 60_000:
                    MTTA_mean = MTTA_factor * 5.4 #mean_processing_time[wc]
                    MTBA_mean = MTTA_mean / Ag - MTTA_mean

                    MTTA = random.uniform(0, 2*MTTA_mean)
                    MTBA = random.expovariate(1 / MTBA_mean)

                    machine_breakdown_times.append(MTBA)
                    machine_repair_times.append(MTTA)
                breakdowntimes.append(machine_breakdown_times)
                repairtimes.append(machine_repair_times)
        return breakdowntimes, repairtimes

    def __init__(self):
        self.QueuesWC = {jj: [] for jj in noOfWC}  # Can be used to keep track of Queue Lenghts
        self.scheduleWC = {ii: [] for ii in noOfWC}  # Used to keep track of the schedule
        self.makespanWC = {ii: np.zeros(machinesPerWC[ii]) for ii in
                           noOfWC}  # Keeps track of the makespan of each machine
        self.last_job_WC = {ii: np.zeros(machinesPerWC[ii]) for ii in
                            noOfWC}  # Keeps track of which job was last in the machine

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


    def reset(self,info):
        self.env = Environment()
        self.machine_per_wc = {(ii, jj): Store(self.env) for jj in noOfWC for ii in
                               range(machinesPerWC[jj])}  # Used to store jobs in a machine
        self.storeWC = {ii: FilterStore(self.env) for ii in noOfWC}  # Used to store jobs in a JPA
        self.condition_flag = {(ii, jj): simpy.Event(self.env) for jj in noOfWC for ii in range(machinesPerWC[jj])}  # An event which keeps track if a machine has had a job inserted into it if it previously had no job
        self.QueuesWC = {jj: [] for jj in noOfWC}  # Can be used to keep track of Queue Lenghts
        self.scheduleWC = {ii: [] for ii in noOfWC}  # Used to keep track of the schedule
        self.makespanWC = {ii: np.zeros(machinesPerWC[ii]) for ii in
                           noOfWC}  # Keeps track of the makespan of each machine
        self.last_job_WC = {ii: np.zeros(machinesPerWC[ii]) for ii in
                            noOfWC}  # Keeps track of which job was last in the machine

        self.flowtime = []  # Jobs flowtime
        self.tardiness = []  # Jobs tardiness
        self.running_flowtime = [[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]]],[[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]]],[[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]]]
        self.machinelog = []
        self.cfptime = []
        self.rewards = []
        self.flow_time_estimate_input = []
        self.flow_time_estimate_output = []
        self.tardiness_estimate_output = []
        self.arrivaltimes = []
        self.interarrivaltimes = []
        self.WIP = 0  # Current WIP of the system
        self.early_termination = 0  # Whether the simulation terminated earlier
        self.utilization = {(ii, jj): 0 for jj in noOfWC for ii in range(machinesPerWC[jj])}
        self.finish_time = 0  # Finishing time of the system
        self.totalWIP = []  # Keeps track of the total WIP of the system
        self.cfp_tardiness = 0
        self.last_repair = [0,0,0]
        self.machines = [[],[],[]]

        self.test_weights = info.weights
        self.arrival_rate = info.arrival_time
        self.ddt = info.due_date_tightness

        self.bids = []  # Keeps track of the bids
        self.priority = []  # Keeps track of the job priorities
        self.start_time = 0  # Starting time of the simulation
        self.max_job = info.max_job
        self.min_job = info.min_job
        self.max_wip = info.max_wip
        self.normalization = info.normalization
        self.items_to_bid = self.env.event()
        repair_time_ind = 1
        breakdown_prob = 0.01
        self.time_to_failure, self.time_to_repair = self.get_breakdown_times(repair_time_ind, breakdown_prob)

        self.env.process(self.source(0, info.arrival_time, info.due_date_tightness, self.min_job))

        for wc in range(len(machinesPerWC)):
            store = self.storeWC[wc]

            self.env.process(self.cfp_wc_trigger(store))
            #self.env.process(self.cfp_wc(self.machine_per_wc, store, wc, self.normalization))

            for ii in range(machinesPerWC[wc]):
                machine = self.machine_per_wc[(ii, wc)]
                self.machines[ii] = self.Machine(self, wc + 1, machine_number_WC[wc][ii], self.test_weights, self.last_job_WC[wc], machine,
                               self.makespanWC[wc],self.min_job, self.max_job, self.normalization, self.max_wip, self.time_to_failure, self.time_to_repair)
        self.end_event = self.env.event()
        self.env.run(until=self.items_to_bid)

        observation = self.get_environment_state(self.storeWC[0],self.machine_per_wc)
        return observation

class New_Job:
    """ This class is used to create a new job. It contains information
    such as processing time, due date, number of operations etc."""

    def __init__(self, name, env, number1, dueDateTightness):
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
        for ii in range(self.numberOfOperations):
            meanPT = processingTimes[self.type - 1][ii]
            self.processingTime[ii] = meanPT
            self.dueDate[ii + 1] = self.dueDate[ii] + self.processingTime[ii] * dueDateTightness

class Info:
    def __init__(self,setting,load,test_weights):
        utilization = [85, 90, 95]
        arrival_time = [2.118, 2.0, 1.895]
        due_date_settings = [4, 4, 4]
        normalization = [[14, 30, 3.5, 4, -600, 23],[14, 30, 3.5, 4, -600, 23],[14, 30, 3.5, 4, -600, 23]]
        min_jobs = [2499, 2499, 2499]
        max_jobs = [4499, 4499, 4499]
        max_wips = [150, 150, 250]
        setting_choice = setting
        if load is True:
            df = pd.read_csv('./DQNRuns/Run-weights-' + str(utilization[setting_choice]) + '-4-1000-[7, 7]-[3, 3].csv', header=None)
            self.weights = df.values.tolist()
        else:
            self.weights = test_weights
        self.min_job = min_jobs[setting_choice]
        self.max_job = max_jobs[setting_choice]
        self.max_wip = max_wips[setting_choice]
        self.arrival_time = arrival_time[setting_choice]
        self.due_date_tightness = due_date_settings[setting_choice]
        self.next_job = []
        self.normalization = normalization[setting_choice]


def get_objectives(job_shop, min_job, max_job, early_termination):
    """This function gathers numerous results from a simulation run"""
    no_tardy_jobs_p1 = 0
    no_tardy_jobs_p2 = 0
    no_tardy_jobs_p3 = 0
    total_p1 = 0
    total_p2 = 0
    total_p3 = 0
    early_term = 0
    if early_termination == 1:
        early_term += 1
        makespan = job_shop.finish_time - job_shop.start_time
        flow_time = np.nanmean(job_shop.flowtime[min_job:max_job]) + 10_000 - np.count_nonzero(
            job_shop.flowtime[min_job:max_job])
        mean_tardiness = np.nanmean(np.nonzero(job_shop.tardiness[min_job:max_job])) + 10_000 - np.count_nonzero(
            job_shop.flowtime[min_job:max_job])
        max_tardiness = np.nanmax(job_shop.tardiness[min_job:max_job])
        for ii in range(min_job, len(job_shop.tardiness)):
            if job_shop.priority[ii] == 1:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p1 += 1
                total_p1 += 1
            elif job_shop.priority[ii] == 3:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p2 += 1
                total_p2 += 1
            elif job_shop.priority[ii] == 10:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p3 += 1
                total_p3 += 1
        # WIP Level
        mean_WIP = np.mean(job_shop.totalWIP)
    else:
        makespan = job_shop.finish_time - job_shop.start_time
        # Mean Flow Time
        flow_time = np.nanmean(job_shop.flowtime[min_job:max_job])
        # Mean Tardiness
        mean_tardiness = np.nanmean(job_shop.tardiness[min_job:max_job])
        # Max Tardiness
        max_tardiness = max(job_shop.tardiness[min_job:max_job])
        # print(len(job_shop.priority))
        # No of Tardy Jobs
        for ii in range(min_job, max_job):
            if job_shop.priority[ii] == 1:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p1 += 1
                total_p1 += 1
            elif job_shop.priority[ii] == 3:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p2 += 1
                total_p2 += 1
            elif job_shop.priority[ii] == 10:
                if job_shop.tardiness[ii] > 0:
                    no_tardy_jobs_p3 += 1
                total_p3 += 1
        # WIP Level
        mean_WIP = np.mean(job_shop.totalWIP)

    return makespan, flow_time, mean_tardiness, max_tardiness, no_tardy_jobs_p1 / total_p1, no_tardy_jobs_p2 / total_p2, no_tardy_jobs_p3 / total_p2, mean_WIP, early_term
#
#
# def do_simulation_with_weights(mean_weight_new, arrivalMean, due_date_tightness, min_job, max_job,
#                                normalization, max_wip, iter1):
#     """ This runs a single simulation"""
#     random.seed(iter1)
#
#     env = Environment()  # Create Environment
#     job_shop = jobShop(env, mean_weight_new)  # Initiate the job shop
#     env.process(source(env, 0, arrivalMean, job_shop, due_date_tightness,
#                        min_job))  # Starts the source (Job Release Agent)
#
#     for wc in range(len(machinesPerWC)):
#         last_job = job_shop.last_job_WC[wc]
#         makespanWC = job_shop.makespanWC[wc]
#         store = job_shop.storeWC[wc]
#
#         env.process(
#             cfp_wc(env, job_shop.machine_per_wc, store, job_shop, wc, normalization))
#
#         for ii in range(machinesPerWC[wc]):
#             machine = job_shop.machine_per_wc[(ii, wc)]
#
#             env.process(
#                 machine_processing(job_shop, wc + 1, machine_number_WC[wc][ii], env, mean_weight_new, last_job,
#                                    machine, makespanWC, min_job, max_job, normalization, max_wip))
#     job_shop.end_event = env.event()
#
#     env.run(until=job_shop.end_event)  # Run the simulation until the end event gets triggered
#
#     makespan, flow_time, mean_tardiness, max_tardiness, no_tardy_jobs_p1, no_tardy_jobs_p2, no_tardy_jobs_p3, mean_WIP, early_term = get_objectives(
#         job_shop, min_job, max_job, job_shop.early_termination)  # Gather all results
#
#     return makespan, flow_time, mean_tardiness, max_tardiness, no_tardy_jobs_p1, no_tardy_jobs_p2, no_tardy_jobs_p3, mean_WIP, early_term
#
# def arrival_calc(utilization):
#     arrival_rate = 5.4/(3*utilization)
#     return arrival_rate
#
#
#
# if __name__ == '__main__':
#     min_jobs = [499, 999, 1499]  # Minimum number of jobs in order te reach steady state
#     max_jobs = [2499, 2999, 3499]  # Maxmimum number of jobs to collect information from
#     wip_max = [150, 200, 300]  # Maxmimum WIP allowed in the system
#
#     utilization = [85, 90, 95]
#     due_date_settings = [4, 4, 4]
#
#     normaliziation = [[-75, 150, -8, 12, -75, 150], [-30, 150, -3, 12, -30, 150], [-200, 150, -15, 12, -200, 150],
#                       [-75, 150, -6, 12, -75, 150], [-300, 150, -35, 12, -300, 150],
#                       [-150, 150, -15, 12, -150, 150]]  # Normalization ranges needed for the bidding
#
#     final_obj = []
#     final_std = []
#
#     no_runs = 1
#     no_processes = 1  # Change dependent on number of threads computer has, be sure to leave 1 thread remaining
#     final_result = np.zeros((no_runs, 9))
#     results = []
#
#     for i in range(len(utilization)):
#         arrival_rate = arrival_calc(utilization[i])
#         str1 = "./DQNRuns/Run-weights-" + str(utilization[i]) + ".csv"
#         df = pd.read_csv(str1, header=None)
#         weights = df.values.tolist()
#         print("Current run is: " + str(utilization[i]) + "-" + str(due_date_settings[i]))
#         obj = np.zeros(no_runs)
#         for j in range(int(no_runs / no_processes)):
#             jobshop_pool = Pool(processes=no_processes)
#             seeds = range(j * no_processes, j * no_processes + no_processes)
#             func1 = partial(do_simulation_with_weights, weights, arrival_rate, due_date_settings[i],
#                             min_jobs[i], max_jobs[i], normaliziation[i], wip_max[i])
#             makespan_per_seed = jobshop_pool.map(func1, seeds)
#             print(makespan_per_seed)
#             for h, o in itertools.product(range(no_processes), range(9)):
#                 final_result[h + j * no_processes][o] = makespan_per_seed[h][o]
#
#         results.append(list(np.mean(final_result, axis=0)))
#     print(results)
#
#     results = pd.DataFrame(results,
#                            columns=['Makespan', 'Mean Flow Time', 'Mean Weighted Tardiness', 'Max Weighted Tardiness',
#                                     'No. Tardy Jobs P1', 'No. Tardy Jobs P2', 'No. Tardy Jobs P3', 'Mean WIP',
#                                     'Early_Term'])
#     file_name = f"Results/Custom_1.csv"
#     results.to_csv(file_name)

# Importing the required libraries
from argparse import ArgumentParser
import time as t
import os
import sys
import psutil
import GPUtil
import numpy as np
import matplotlib.pyplot as plt
import json
from os.path import join
from datetime import datetime

parser = ArgumentParser(description='program for running other processes')

# arguments that is needed for every type
parser.add_argument('-p', '--pid', type=int, help='specify the pid', required=True)
parser.add_argument('-log', '--logPath', help='path to the log file')
args = parser.parse_args()



# Creating an almost infinite for loop to monitor the details continuously
def monitoring(pid, log_path):
    TIME = []
    CPU = []
    RAM = []
    GPU = []
    GPU_MEM = []
    monitor_log = join(log_path, 'monitor_log.json')
    figure_save = join(log_path, 'figure.png')
    time_start = t.time()
    total_ram = psutil.virtual_memory().total/1024/1024
    p = psutil.Process(pid)
    gpu_list = []
    # gpu_mem_total = gpu_list[0].memoryTotal
    # for i in range(100000000):
    while(True):
        # Obtaining all the essential details
        time_stop = t.time()
        time = time_stop - time_start
        print("Time run: ", time)
        cpu_usage = p.cpu_percent(interval=1)/psutil.cpu_count()
        mem_usage = p.memory_percent()/100*total_ram
        gpu_list = GPUtil.getGPUs()
        gpu_usage = gpu_list[0].load*100
        gpu_mem_usage = gpu_list[0].memoryUsed
        print(cpu_usage)
        print(mem_usage)
        print("CPU used :",cpu_usage, "%")
        print("GPU used :",gpu_usage, "%")
        print("CPU mem used :",mem_usage, "MB")
        print("GPU mem used :",gpu_mem_usage, "MB")
        print("Total mem: ",psutil.virtual_memory().total/1024/1024)
                # Obtaining the GPU details
        # GPUtil.showUtilization()



        TIME.append(time)
        CPU.append(cpu_usage)
        RAM.append(mem_usage)
        GPU.append(gpu_usage)
        GPU_MEM.append(gpu_mem_usage)

        data = {
            'time'   : TIME,
            'cpu'    : CPU,
            'ram'    : RAM,
            'gpu'    : GPU,
            'gpu_mem': GPU_MEM,
        }
        with open(monitor_log,'w') as file: 
            json.dump(data, file)
        # Creating the scatter plot
        a1 = plt.subplot(2,1,1)
        a1.scatter(time, cpu_usage, color = "red", linewidths = 0.2)
        a1.scatter(time, gpu_usage, color = "green", linewidths = 0.2)
        a1.set_ylabel("Percent(%)")
        a1.legend(["CPU","GPU"], loc ="best")


        a2 = plt.subplot(2,1,2)
        a2.scatter(time, mem_usage, color = "blue", linewidths = 0.2)
        a2.scatter(time, gpu_mem_usage, color = "orange", linewidths = 0.2)
        a2.set_ylabel("Usage (MB)")
        a2.legend(["CPU Memory", "GPU Memory"], loc ="best")

        plt.suptitle("Resource Monitoring")
        #plt.legend(["CPU", "Memory", "GPU", "GPU Memory"], loc ="best")
        plt.xlabel("Time(s)")
        #plt.ylabel("Percent(%)")
        plt.pause(0.05)
        plt.savefig(figure_save, dpi=200)

        # print("psutil.cpu_percent(interval=2) = %s" % (p.cpu_percent(interval=2),))

        # print("psutil.cpu_percent(interval=2) = %s" % (p.cpu_percent(interval=2),))

        # print("The number of CPUs is : %s" % (p.cpu_num(), ))

        # print("The CPU utilization of all the CPUs is: %s" % (p.cpu_percent(interval=2, percpu=True), ))

        print("------ PID ------- : ",pid)
    # Plotting the information
    # fig, ax = plt.subplots()
    # fig = plt.gcf().savefig(figure_save)
    # plt.show()
    # plt.savefig(figure_save)
    return 0

monitoring(args.pid, args.logPath)

import psutil
import time
import numpy as np
import gpustat


class MonitorResources:
    def __init__(self, debug=False):
        self.debug = debug
        self.initial_memory_usage = psutil.virtual_memory()
        self.initial_cpu_usage = psutil.cpu_percent(interval=1)
        self.cpu_usage = []
        self.gpu_usage = []
        self.memory_usage = []
        self.start_monitoring()

    def start_monitoring(self):
        while True:
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent()
            gpu_usage = gpustat.GPUStatCollection.new_query()[0].utilization

            self.cpu_usage.append(cpu_usage)
            self.gpu_usage.append(gpu_usage)
            self.memory_usage.append(memory_usage)

            if self.debug:
                print(f"CPU Usage: {cpu_usage}% ;;;;;;;;; GPU Usage: {gpu_usage}% ;;;;;;;;; Memory Usage: {memory_usage}%")

            print(f'CPU avg usage: {np.average(self.cpu_usage)} ;;;;;;;;;;; GPU avg usage: {np.average(self.gpu_usage)} ;;;;;;;;;;; Memory avg/max usage: {np.average(self.memory_usage)}/{np.max(self.memory_usage)} ')
            time.sleep(1)

    def stop_and_get_average(self):
        return np.average(self.cpu_usage), np.average(self.memory_usage)


monitor_resource = MonitorResources(debug=False)


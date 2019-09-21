#!/usr/bin/env python
import multigpuexec
import time
import os

gpus = range(0, 1)
batchsizes = [7, 8, 9] + range(10, 200, 10) + range(200, 501, 50)
# CHANGE HOST NAME
host = "mouse"

# os.chdir("/model")
gpu_model = multigpuexec.getGPUinfo(0, query="name", get_cuda=False).strip().replace(" ", "")
print gpu_model

# imgsize = 224
command = "python benchmark.py --testVGG16 --no_timeline --saveto results --iter_benchmark 10 --imgsize 32 --keras --device {} --comment {}".format(
    gpu_model, host)
tasks = []

for batch in batchsizes:
    command_pars = command + " --batchsize {}".format(batch)
    task = {"comm": command_pars}
    tasks.append(task)

print "Have", len(tasks), "tasks"
print "Running in {}".format(os.getcwd())
gpu = -1
for i in range(0, len(tasks)):
    gpu = multigpuexec.getNextFreeGPU(gpus, start=gpu + 1, c=3, d=1)
    multigpuexec.runTask(tasks[i], gpu, delay=0)
    print "{}/{} tasks".format(i + 1, len(tasks))
    time.sleep(1)

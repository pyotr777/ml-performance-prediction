# MutliGPU series execution support
# 2018 (C) Peter Bryzgalov @ CHITECH Stair Lab

from __future__ import print_function
import subprocess
import re
import time
import os


def message(s, col=112):
    print("\033[38;5;{}m{}\033[0m".format(col, s))


maxtemp = 60
running_pids = {}


# Returns True if GPU #i is not used.
# Uses nvidia-smi command to monitor GPU SM usage.
def GPUisFree(i, c=1, d=1, mode="dmon", debug=False):
    global running_pids
    gpu_free = False
    if mode == "dmon":
        # Doesn't work in (AWS) VMs
        command = "nvidia-smi dmon -c {} -d {} -i {} -s u".format(c, d, i)
        if debug:
            print(command)
        out_pattern = re.compile(r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")  # dmon
    else:
        # Doesn't work in docker containers
        # Doesn't work when GPU card used by X Window server
        command = "nvidia-smi pmon -c {} -d {} -i {} -s u".format(c, d, i)
        out_pattern = re.compile(r"^\s+(\d+)\s+([0-9\-]+)\s+([CG\-])\s+([0-9\-]+)\s")  # pmon
    proc = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=1, shell=False)
    u = 0
    for line in iter(proc.stdout.readline, b''):
        line = line.decode('utf-8')
        if debug:
            print(line.strip(" "), end="")
        m = out_pattern.search(line)
        if m:
            print(".", end="")
            uplus = 0
            try:
                uplus = int(m.group(2))
            except ValueError:
                pass
            u += uplus
    if u < 11:
        gpu_free = True
    else:
        if mode == "dmon":
            print("busy {:.0f}%".format(float(u) / float(c)))
        else:
            print("busy")

    # Check Temp
    temp_norm = False
    if gpu_free and mode == "dmon":
        command = "nvidia-smi dmon -s p -d 1 -c 1 -i {}".format(i)
        out_pattern = re.compile(r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+.*")  # dmon
        proc = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=1, shell=False)
        temp = 0
        for line in iter(proc.stdout.readline, b''):
            line = line.decode('utf-8')
            if debug:
                print(line.strip(" "), end="")
            m = out_pattern.search(line)
            if m:
                print(".", end="")
                t = 0
                try:
                    t = int(m.group(3))
                except ValueError:
                    pass
                if debug:
                    print("{}C".format(t))
                if t > temp:
                    temp = t

        if temp < maxtemp:
            temp_norm = True
        else:
            message("Overheat GPU{}: {}C".format(i, temp), 196)
            gpu_free = False
            return gpu_free

    # Check PIDs
    if gpu_free and temp_norm:
        if debug:
            print("GPU looks free. PIDS:", [pid for _, pid in running_pids.iteritems()])
        if i in running_pids:
            pid = running_pids[i]
            # Check if process still alive
            try:
                os.kill(pid, 0)
                # If no exception, process is still running
                if debug:
                    print("Process {} on {} is running".format(pid, i))
                print("busy")
                gpu_free = False
            except Exception as e:
                # Process died
                # Remove pid from running_pids
                if debug:
                    print("Exception on {} : {}".format(pid, e))
                del running_pids[i]
    if gpu_free:
        print("free")
    return gpu_free


# Returns GPU info
def getGPUinfo(i, query="name,memory.total,memory.free,pstate", get_cuda=True):
    command = "nvidia-smi -i {} --query-gpu={} --format=csv,noheader".format(i, query)
    proc = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=1, shell=False)
    output = ""
    for line in iter(proc.stdout.readline, b''):
        line = line.decode('utf-8')
        output += line
    if get_cuda:
        command = "./get_nvidia_versions.sh"
        proc = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=1, shell=False)
        for line in iter(proc.stdout.readline, b''):
            line = line.decode('utf-8')
            output += line
    return output


# Returns CPU models and frequencies
def getCPUinfo():
    command = "lscpu"
    patterns = [r"^CPU\(s\):", r"^Model name:", r"^CPU MHz:", r"^CPU max MHz:"]
    proc = subprocess.Popen(command.split(" "),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            bufsize=1,
                            shell=False)
    output = ""
    for line in iter(proc.stdout.readline, b''):
        line = line.decode('utf-8')
        for pattern in patterns:
            if re.search(pattern, line):
                # output += line.split(":")[1].strip(" \n") + " "
                output += line
                continue
    return output


# Returns number of a free GPU.
# gpus  -- GPU number or list of numbers.
# start -- number of GPU to start with.
def getNextFreeGPU(gpus, start=-1, c=4, d=1, nvsmi=False, mode="dmon", debug=False):
    if not isinstance(gpus, list):
        gpus = [gpus]
    if start > gpus[-1]:
        # Rewind to GPU 0
        start = 0
    while True:
        for i in range(0, len(gpus)):
            gpu = gpus[i]
            if gpu < start:
                continue
            print("checking GPU", gpu, end="")
            if GPUisFree(gpu, c=c, d=d, mode=mode, debug=debug):
                return gpu
            time.sleep(d)
            start = -1  # Next loop check from 1


# Runs a task on specified GPU
def runTaskContainer(task, gpu, verbose=False):
    f = open(task["logfile"], "ab")
    # f.write("gpu"+str(gpu)+"\n")
    command = task["comm"]
    # command = "python --version"
    # IMPORTANT: remote double spaces or they will become empty arguments!
    command = re.sub(' \s+', ' ', command).strip()
    command = "NV_GPU=" + str(gpu) + " ./run_container.sh " + command
    print("Starting ", command)
    if not verbose:
        pid = subprocess.Popen(command, stdout=f, stderr=f, bufsize=1, shell=True).pid
        print(pid)
    else:
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, shell=True)
        for line in iter(p.stdout.readline, ''):
            print(line.rstrip())
            f.write(line)
    f.close()


# Runs a task on specified GPU
def runTask(task, gpu, nvsmi=False, delay=3, interval=2, debug=False):
    global running_pids

    # f.write("gpu"+str(gpu)+"\n")
    # Set GPU for execution with env var CUDA_VISIBLE_DEVICES
    my_env = os.environ.copy()
    my_env[b"CUDA_VISIBLE_DEVICES"] = str(gpu)
    my_env[b"NVIDIA_VISIBLE_DEVICES"] = str(gpu)
    if debug:
        for k in my_env.keys():
            print("{}={}".format(k, my_env[k]))
    command = task["comm"]
    # IMPORTANT: remove double spaces or they will become empty arguments!
    command = re.sub(' \s+', ' ', command).strip()
    # Insert GPU number into command instead of gpu_num pattern
    command = command.replace("gpu_num", str(gpu))
    print("Starting on GPU{}".format(gpu))
    message(command)
    if "logfile" in task:
        f = open(task["logfile"], "ab")
        pid = subprocess.Popen(command.split(" "), stdout=f, stderr=subprocess.STDOUT, bufsize=1, env=my_env).pid
    else:
        proc = subprocess.Popen(command.split(" "), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=1, env=my_env)
        pid = proc.pid
        for line in iter(proc.stdout.readline, b''):
            line = line.decode('utf-8')
            print(line, end="")

    print(pid)
    # Save pid to runnning processes dictionary
    running_pids[gpu] = pid

    if (nvsmi):
        # Save memory usage info
        # Wait before process starts using GPU
        sampling_rate = 50  # msec
        sampling_period = interval  # sec
        time.sleep(delay)
        # Save stdout to logfile
        logfile = task["logfile"] + ".nvsmi"
        fl = open(logfile, "w")
        command = "nvidia-smi -i {} -lms {} --query-gpu=timestamp,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits".format(
            gpu, sampling_rate)
        p = subprocess.Popen(command.split(" "), stdout=fl, stderr=subprocess.STDOUT, bufsize=1, shell=False)
        time.sleep(sampling_period)
        p.kill()
        fl.close()

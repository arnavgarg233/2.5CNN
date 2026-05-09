import psutil
import getpass
import subprocess
import re

def get_gpu_process_ids():
    try:
        nvidia_smi_output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        # This will return a string where each line contains a PID
        pids = nvidia_smi_output.strip().split('\n')
        return [int(pid) for pid in pids if pid.isdigit()]
    except subprocess.CalledProcessError:
        print("Unable to run nvidia-smi.")
        return []

gpu_pids = get_gpu_process_ids()

def get_user_gpu_processes(user, gpu_pids):
    user_gpu_processes = []
    all_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username']):
        if proc.info['pid'] in gpu_pids:
            all_processes.append(proc.info)
            if proc.info['username'] == user:
                user_gpu_processes.append(proc.info)
    print(all_processes)
    return user_gpu_processes

# Get the current user's username
current_user = getpass.getuser()

# Filter GPU processes by user
user_gpu_processes = get_user_gpu_processes(current_user, gpu_pids)
for proc in user_gpu_processes:
    print(f"PID: {proc['pid']}, Name: {proc['name']}")

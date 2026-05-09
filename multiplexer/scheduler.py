import subprocess
import time
import re
import os
import sys 

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)


from utils import load_config, exp_name_from_config
def get_next_job_id(job_id_file):
    try:
        with open(job_id_file, 'r') as file:
            job_id = int(file.read().strip()) + 1
    except FileNotFoundError:
        job_id = 1  # Start from 1 if the file does not exist

    with open(job_id_file, 'w') as file:
        file.write(str(job_id))

    return job_id

def get_next_dont_increment(job_id_file):
    try:
        with open(job_id_file, 'r') as file:
            job_id = int(file.read().strip())
    except FileNotFoundError:
        job_id = 1  # Start from 1 if the file does not exist

    with open(job_id_file, 'w') as file:
        file.write(str(job_id))

    return job_id


def parse_nvidia_smi():
    try:
        # Run nvidia-smi command
        nvidia_smi_output = subprocess.check_output(["nvidia-smi"], text=True)
        # Use regular expression to extract memory usage
        # This regex looks for patterns like "  6000MiB /  8000MiB"
        regex_pattern = r'(\d+MiB) / \s*(\d+MiB)'
        matches = re.findall(regex_pattern, nvidia_smi_output)

        # Process the matches to get a list of [used_memory, total_memory]
        gpu_memory = [{"used_memory": match[0], "total_memory": match[1]} for match in matches]
        gpu_memory = {
            i: int(memory['used_memory'][:-3]) for i, memory in enumerate(gpu_memory)
        }
        return gpu_memory
    except subprocess.CalledProcessError as e:
        print("Unable to run nvidia-smi.")
        return None

def read_job_queue(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read().splitlines()
    except FileNotFoundError:
        return []  # Return an empty list if file does not exis t

def write_job_queue(file_path, jobs):
    with open(file_path, 'w') as file:
        file.write('\n'.join(jobs))


def append_job(file_path, job, job_id_file, config_path):
    # Get the next job ID and create a branch
    job_id = get_next_job_id(job_id_file)
    exp_name = exp_name_from_config(config_path=config_path)
    branch_name = f"versions/{job_id}/{exp_name}"
    subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
    subprocess.run(['git', 'push', 'versions', branch_name], check=True)


    # Append the job command along with the branch name to the job file
    job_entry = f"{branch_name}:{job}"
    jobs = read_job_queue(file_path)
    jobs.append(job_entry)
    write_job_queue(file_path, jobs)

    print(f"Appended job: {job} on branch {branch_name}")
    subprocess.run(['git', 'checkout', 'master'], check=True)

def enqueue_from_file(filename):
    # Open and read the file
    with open(filename, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if line:  # Check if the line is not empty
            # Construct the command
            job_command = f"python launch.py --config {line} --force"
            command = [
                'python', 'multiplexer/scheduler.py', 'append',
                '--file_path', 'multiplexer/job_queue.txt',
                '--job', job_command,
                '--config_path', line
            ]

            # Execute the command
            subprocess.run(command, check=True)

            print(f"Enqueued job for config: {line}")

def launch_using_sh(job_command, gpu_index, git_branch):
    script_path = 'multiplexer/launch_gpu.sh'
    command = ['bash', script_path, job_command, str(gpu_index), git_branch]
    subprocess.run(command, check=True)
    print(f"Launched job '{job_command}' on GPU {gpu_index} using branch '{git_branch}'")

def launch_job_in_screen(job_command, gpu_index, git_branch, session_name):
    # Construct the command for the script
    script_path = 'multiplexer/launch_gpu.sh'
    script_command = f"bash {script_path} '{job_command}' {gpu_index} {git_branch}"

    # Create a new screen session and run the script
    subprocess.run(['screen', '-dmS', session_name, script_command])

    print(f"Launched job '{job_command}' on GPU {gpu_index} in screen session '{session_name}'")

def launch_job_in_tmux(job_command, gpu_index, git_branch, session_name):
    script_path = 'multiplexer/launch_gpu.sh'
    script_command = f"bash {script_path} '{job_command}' {gpu_index} {git_branch}"

    # Create a new tmux session and run the script
    subprocess.run(['tmux', 'new-session', '-d', '-s', session_name, script_command])

    print(f"Launched job '{job_command}' on GPU {gpu_index} in tmux session '{session_name}'")


def launch_job_on_gpu(job_command, gpu_index, git_branch):
    subprocess.run(['git', 'fetch', 'versions', git_branch], check=True)
    subprocess.run(['git', 'checkout', git_branch], check=True)
    # Set the environment variable to use a specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    # Launch the job (example using subprocess)
    subprocess.Popen(job_command, shell=True)
    print(f"Launched job on GPU {gpu_index}")
    time.sleep(5)

    # Return to master branch
    subprocess.run(['git', 'checkout', 'master'], check=True)

def schedule_jobs(file_path):
    while True:
        jobs = read_job_queue(file_path)
        if not jobs:
            break  # Exit if there are no jobs

        gpu_memory_list = parse_nvidia_smi()
        if gpu_memory_list is None:
            break  # Exit if unable to read GPU status
        all_busy = True
        for gpu_index, used_memory in gpu_memory_list.items():
            if used_memory < 500:  # Check if less than 300MiB is used
                job_command = jobs.pop(0)  # Get the first job from the list
                write_job_queue(file_path, jobs)
                branch_name, job_command = job_command.split(':')
                print(f"Launching {job_command} on GPU {gpu_index}")
                # launch_job_on_gpu(job_command, gpu_index, branch_name)
                # launch_using_sh(job_command, gpu_index, branch_name)
                # launch_job_in_screen(job_command, gpu_index, branch_name, f'gpu_{gpu_index}')
                launch_job_in_tmux(job_command, gpu_index, branch_name, f'gpu_{gpu_index}')
                all_busy = False
                break  # Break to restart the loop and recheck GPU statuses
        if all_busy:
            time.sleep(120)  # Wait for 2 minutes before checking again
        else:
            time.sleep(20)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Job Scheduler CLI")
    subparsers = parser.add_subparsers(dest='command')

    # Parser for append_job
    append_parser = subparsers.add_parser('append', help='Append a new job')
    append_parser.add_argument('--file_path', type=str, help='Path to the job queue file')
    append_parser.add_argument('--job', type=str, help='The job command to append', default="multiplexer/job_queue.txt")
    append_parser.add_argument('--config_path', type=str, help='Path to config', required=True)
    append_parser.add_argument('--job_id_file', type=str, help='Path to the job ID file', default="multiplexer/job_id.txt")

    # Parser for schedule_jobs
    schedule_parser = subparsers.add_parser('execute', help='Schedule jobs')
    schedule_parser.add_argument('--file_path', type=str, help='Path to the job queue file', default="multiplexer/job_queue.txt")

    # parser for enqueue_from_file
    enqueue_parser = subparsers.add_parser('enqueue', help='Enqueue jobs from a file')
    enqueue_parser.add_argument('--file_path', type=str, help='File containing experiment setups', default="multiplexer/enqueue.txt")

    args = parser.parse_args()
    return args

def cli():
    args = main()

    if args.command == 'append':
        append_job(args.file_path, args.job, args.job_id_file, args.config_path)
    elif args.command == 'execute':
        schedule_jobs(args.file_path)
    elif args.command == 'enqueue':
        enqueue_from_file(args.file_path)
    else:
        print("Invalid command. Use 'append' or 'schedule'.")


if __name__ == '__main__':
    cli()


# if __name__ == '__main__':
#     # schedule_jobs('job_queue.txt')
#     time.sleep(5)
#     append_job('job_queue.txt', 'python launch.py --config configs/3dcnn.yaml --force --exp_name 3d-cnn', '3d-cnn', 'job_id.txt')
import subprocess

# Your Python interpreter and script paths
interpreter = "/home/jypaulsung/Sapien/.venv/bin/python"
script_path = "/home/jypaulsung/Sapien/ArrayCan/can_detection/starting_coordinates.py"

# Number of iterations for each seed
num_iterations = 10
timeout_sec = 10 # Automatically kill the process after 10 seconds

for seed in range(11, 21):
    for i in range(num_iterations):
        print(f"Running iteration {i} with seed {seed}")
        try:
            subprocess.run(
                [interpreter, script_path, f"--seed={seed}"],
                timeout=timeout_sec
            )
        except subprocess.TimeoutExpired:
            print(f"Iteration {i} timed out after {timeout_sec} seconds.")
        except subprocess.CalledProcessError as e:
            print(f"Script failed for iteration {i} with error: {e}")


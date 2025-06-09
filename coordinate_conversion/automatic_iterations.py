import subprocess

# Your Python interpreter and script paths
interpreter = "/home/jypaulsung/Sapien/.venv/bin/python"
script_path = "/home/jypaulsung/Sapien/ArrayCan/coordinate_conversion/destination_coords.py"

# Number of iterations for each seed
num_iterations = 10
timeout_sec = 5 # Automatically kill the process after 5 seconds

for seed in range(2, 11):
    print(f"Finding destination coordinates for seed {seed}")
    try:
        subprocess.run(
            [interpreter, script_path, f"--seed={seed}"],
            timeout=timeout_sec
        )
    except subprocess.TimeoutExpired:
        print(f"Finding destination coordinates for seed {seed} timed out after {timeout_sec} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"Script failed for seed {seed} with error: {e}")


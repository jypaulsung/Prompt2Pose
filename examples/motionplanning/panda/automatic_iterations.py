import subprocess

# Your Python interpreter and script paths
interpreter = "/home/jypaulsung/Sapien/.venv/bin/python"
script_path = "/home/jypaulsung/Sapien/.venv/lib/python3.11/site-packages/mani_skill/examples/motionplanning/panda/run.py"

timeout_sec = 30 # Automatically kill the process after 30 seconds
# Environment name for run.py
env = "ArrayCan-v0"

for seed in range(3, 11):
    print(f"Trying motion planning for seed {seed} and saving destination reference coordinates")
    for iter in range(0, 10):
        try:
            subprocess.run(
                [interpreter, script_path, f"-e={env}", f"--seed={seed}", f"--iteration={iter}"],
                timeout=timeout_sec
            )
        except subprocess.TimeoutExpired:
            print(f"Trying motion planning for seed {seed} iteration {iter} timed out after {timeout_sec} seconds.")
        except subprocess.CalledProcessError as e:
            print(f"Script failed for seed {seed} iteration {iter} with error: {e}")


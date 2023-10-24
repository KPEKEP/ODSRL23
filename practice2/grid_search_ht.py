import wandb
import yaml
import argparse

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description="Grid Search Sweep Script using Weights & Biases")
parser.add_argument("--training_script", type=str, default="training_script.py", help="Filename of the training script.")
parser.add_argument("--config_file", type=str, default="sweep_config.yaml", help="Filename of the sweep configuration YAML file.")
args = parser.parse_args()

# Load sweep configuration from the YAML file
with open(args.config_file, "r") as file:
    sweep_config = yaml.safe_load(file)
    
# Initialize the sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=sweep_config['name'])

# Define the function to run the training script
def main_function():
    import subprocess
    import os

    # Construct the command to run the provided training script
    cmd = f"python {args.training_script}"
    
    # Set up the environment variables for the subprocess
    env = os.environ.copy()
    env["WANDB_SWEEP_ID"] = sweep_id

    # Execute the command
    subprocess.run(cmd, shell=True, env=env)



# Use wandb.agent to run the sweep
wandb.agent(sweep_id, function=main_function)
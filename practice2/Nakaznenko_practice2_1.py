import argparse
import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, TimeLimit, AutoResetWrapper
import wandb
import modeling.agents
from modeling.trainer import Trainer
from modeling.utils import check_same_device
import torch

#import warnings
#warnings.filterwarnings('error')

def get_agent_class(agent_name):
    """Retrieve the agent class based on its name."""
    return getattr(modeling.agents, agent_name)

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train and evaluate a CrossEntropyAgent.")
    parser.add_argument("--agent", type=str, default="CrossEntropyAgent", help="Type of agent to use.")
    parser.add_argument("--env", type=str, default="Taxi-v3", help="Gym environment name.")
    parser.add_argument("--train", action="store_true", help="Flag to train the model.")
    parser.add_argument("--filename", type=str, default=None, help="File name to save/load the model.")
    parser.add_argument("--trajectory_n", type=int, default=2500, help="Number of trajectories.")
    parser.add_argument("--episode_n", type=int, default=1000, help="Number of episodes.")
    parser.add_argument("--gamma_q", type=float, default=0.5, help="Gamma quantile.")
    parser.add_argument("--delta_q", type=float, default=-1, help="Delta quantile.")
    parser.add_argument("--max_trajectory_len", type=int, default=100, help="Maximum trajectory length.")
    parser.add_argument("--render", action="store_true", help="Render the environment during training and evaluation.") 
    parser.add_argument("--project", type=str, default="reinforcement_project", help="Name for the project for wandb")
    parser.add_argument("--wandb", action="store_true", help="Log to wandb")  
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for computation")
    parser.add_argument("--fp32", action="store_true", help="Use Float32 precision")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of environments to run in parallel.")
    parser.add_argument("--exploration", type=float, default=0.5, help="Exploration chance.")    
    parser.add_argument("--gamma_discount", type=float, default=1.0, help="Gamma for discount")
    parser.add_argument("--goal", type=float, default=float("inf"), help="Stop training if mean total reward is above this number upon eval")
    parser.add_argument("--record", action="store_true", help="Record simulation process")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of the batch")    
    parser.add_argument("--fit_attempts", type=int, default=1, help="Number of fit epochs")    

    args = parser.parse_args()
    
    # Check if wandb is initialized (i.e., running a sweep)
    if wandb.run:
        args.wandb = True
        # Override args with values from wandb.config
        args.agent = wandb.config.agent
        args.env = wandb.config.env
        args.trajectory_n = wandb.config.trajectory_n
        args.episode_n = wandb.config.episode_n
        args.gamma_q = wandb.config.gamma_q
        args.delta_q = wandb.config.delta_q
        args.max_trajectory_len = wandb.config.max_trajectory_len
        args.train = wandb.config.train if hasattr(wandb.config, 'train') else args.train
        args.lr = wandb.config.lr
        args.fp32 = wandb.config.fp32
        args.num_proc = wandb.config.num_proc
        args.exploration = wandb.config.exploration
        args.gamma_discount = wandb.config.gamma_discount
        args.goal = wandb.config.goal
        args.batch_size = wandb.config.batch_size
        args.fit_attempts = wandb.config.fit_attempts
        args.record = wandb.config.record
        args.cpu = wandb.config.cpu
        
        if hasattr(wandb.config, 'filename'):
            args.filename = wandb.config.filename
        args.render = wandb.config.render if hasattr(wandb.config, 'render') else args.render
    
    if not args.filename:
        if (not os.path.exists("checkpoints")):
            os.mkdir("checkpoints")
        args.filename = f"checkpoints/{args.agent}.pth"

    if args.record:
        if (not os.path.exists("videos")):
            os.mkdir("videos")        
    
    device = "cuda" if not args.cpu and torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    dtype = torch.float32 if args.fp32 else torch.float64
    torch.set_default_dtype(dtype)

    args.num_proc = max(0, args.num_proc)
    render_mode = "human" if args.render else None
    render_mode = "rgb_array" if args.record else render_mode
    
    def make_env():
        env = AutoResetWrapper(TimeLimit(gym.make(args.env, render_mode=render_mode), max_episode_steps=args.max_trajectory_len))
        if args.record:
            env = RecordVideo(env, "videos", episode_trigger=lambda x: True)
        return env
        
    env = gym.vector.AsyncVectorEnv([make_env] * args.num_proc)
        
    # Check if the script is being run as part of a sweep
    if args.wandb:
        if "WANDB_SWEEP_ID" in os.environ:
            wandb.init()
        else:
            # Initialize from CLI args
            wandb.init(project=args.project)
        
    # Dynamically create the agent based on the provided argument
    AgentClass = get_agent_class(args.agent)
    agent = AgentClass(env, lr=args.lr, device=device, dtype=dtype)
    assert check_same_device(agent), "Mismatch of device of the parameters within the model"
    
    if not args.train:
        agent.load(args.filename, env, device=device)

    trainer = Trainer(env, agent, gamma=args.gamma_discount, device=device, dtype=dtype)

    if args.train:
        trainer.train(args.trajectory_n, args.episode_n, 
                      args.gamma_q, args.delta_q, args.exploration,
                      args.goal, args.batch_size,
                      args.fit_attempts,
                      log=args.wandb)
        agent.save(args.filename)

    # Evaluate
    trainer.eval(args.trajectory_n, args.episode_n, 
                 log=args.wandb)

    if args.wandb:
        wandb.finish()
import argparse
import gym
import numpy as np
from tqdm.auto import tqdm
import time
import logging
import wandb
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FancyInterpolator:
    """Interpolates between two values using a custom function."""

    def __init__(self, A, B):
        """Initialize the interpolator with start value A and end value B."""
        self.A = A
        self.B = B

    def __call__(self, t):
        """Interpolate between A and B using parameter t."""
        if t < 0 or t > 1:
            raise ValueError("Interpolation parameter t should be in the range [0, 1].")
        return self.A * (self.B / self.A) ** (t**(np.cos(np.pi/2 * t**0.2)))

class CrossEntropyAgent:
    """Agent that uses the Cross Entropy method for policy optimization."""

    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
        self.actions = np.arange(self.action_n, dtype=int)
        self.model = np.ones((state_n, action_n)) / action_n

    def get_action(self, state):
        """Sample an action based on the current policy for the given state."""
        action = np.random.choice(self.actions, p=self.model[state])
        return action.item()

    def fit(self, trajectories, laplace_lambda = 0.0, policy_lambda = 1.0):
        """Update the policy based on the provided trajectories."""
        new_model = np.full(self.model.shape, laplace_lambda)

        # Accumulate counts for state-action pairs
        for trajectory in trajectories:
            new_model[trajectory['states'], trajectory['actions']] += 1

        # Normalize the rows of new_model
        row_sums = new_model.sum(axis=1, keepdims=True)
        non_zero_rows = row_sums.squeeze() > 0
        new_model[non_zero_rows] /= row_sums[non_zero_rows]

        # For rows with zero sum, copy from the old model
        zero_rows = np.logical_not(non_zero_rows)
        new_model[zero_rows] = self.model[zero_rows]

        # apply policy smoothing
        self.model = policy_lambda * new_model + (1 - policy_lambda) * self.model

    def save(self, filename):
        """Save the agent's model to a file."""
        np.save(filename, self.model)

    def load(self, filename):
        """Load the agent's model from a file."""
        self.model = np.load(filename)

class Trainer:
    """Trains and evaluates the CrossEntropyAgent in a given environment."""

    def __init__(self, env, agent, render=False, interpolate=False):
        self.env = env
        self.agent = agent
        self.render = render
        self.interpolate = interpolate

    def get_state(self, obs):
        """Convert observation to state."""
        return int(obs)

    def get_trajectory(self, max_trajectory_len):
        """Generate a trajectory using the agent's policy."""
        trajectory = {'states': [], 'actions': [], 'rewards': []}

        obs = self.env.reset()
        state = self.get_state(obs)

        for _ in range(max_trajectory_len):
            trajectory['states'].append(state)

            action = self.agent.get_action(state)
            trajectory['actions'].append(action)

            obs, reward, done, _ = self.env.step(action)
            trajectory['rewards'].append(reward)

            state = self.get_state(obs)

            if self.render:
                self.env.render()
                time.sleep(0.05)
            if done:
                break

        return trajectory

    def train(self, trajectory_n, iteration_n, 
              gamma_q, max_trajectory_len, 
              laplace_f=0.0, policy_f=1.0,
              verbose=True):
        """Train the agent using the Cross Entropy method."""
        traj_len_interp = FancyInterpolator(trajectory_n, trajectory_n * 0.1)
        quant_interp = FancyInterpolator(gamma_q, gamma_q / 100)

        results = []
        for it in tqdm(range(iteration_n), desc="Train"):
            t = it / iteration_n
            max_traj = int(traj_len_interp(t)) if self.interpolate else trajectory_n        
            trajectories = [self.get_trajectory(max_trajectory_len) for _ in range(max_traj)]
            total_rewards = [self.get_total_reward(trajectory) for trajectory in trajectories]
            mean_total_reward = np.mean(total_rewards)
            min_total_reward = np.min(total_rewards)
            max_total_reward = np.max(total_rewards)

            # Log to wandb
            wandb.log({
                "Training/Mean_Total_Reward": mean_total_reward,
                "Training/Min_Total_Reward": min_total_reward,
                "Training/Max_Total_Reward": max_total_reward
            })

            if verbose:
                logger.info(f"{it=} {min_total_reward=} {mean_total_reward=} {max_total_reward=} {traj_len_interp(t)=} {quant_interp(t)=}")

            # Policy 
            q = quant_interp(t) if self.interpolate else gamma_q
            quantile = np.quantile(total_rewards, q)
            elite_trajectories = [trajectory for i, trajectory in enumerate(trajectories) if total_rewards[i] > quantile]

            # Fit agent
            self.agent.fit(elite_trajectories, laplace_f, policy_f)

            results.append({
                'min_total_reward': min_total_reward,
                'mean_total_reward': mean_total_reward,
                'max_total_reward': max_total_reward
            })

        return results

    def eval(self, trajectory_n, iteration_n, max_trajectory_len, verbose=True):
        """Evaluate the agent's performance."""
        results = []
        for it in tqdm(range(iteration_n), desc="Eval"):
            trajectories = [self.get_trajectory(max_trajectory_len) for _ in range(trajectory_n)]
            total_rewards = [self.get_total_reward(trajectory) for trajectory in trajectories]
            mean_total_reward = np.mean(total_rewards)
            min_total_reward = np.min(total_rewards)
            max_total_reward = np.max(total_rewards)

            # Log to wandb
            wandb.log({
                "Evaluation/Mean_Total_Reward": mean_total_reward,
                "Evaluation/Min_Total_Reward": min_total_reward,
                "Evaluation/Max_Total_Reward": max_total_reward
            })

            if verbose:
                logger.info(f"{it=} {mean_total_reward=} {min_total_reward=} {max_total_reward=}")

            results.append({
                'min_total_reward': min_total_reward,
                'mean_total_reward': mean_total_reward,
                'max_total_reward': max_total_reward
            })

        return results

    def get_total_reward(self, trajectory):
        """Calculate the total reward for a trajectory."""
        return np.sum(trajectory['rewards'])

if __name__ == "__main__":
    def main_function():
        # Argument parsing
        parser = argparse.ArgumentParser(description="Train and evaluate a CrossEntropyAgent.")
        parser.add_argument("--env", type=str, default="Taxi-v3", help="Gym environment name.")
        parser.add_argument("--train", action="store_true", help="Flag to train the model.")
        parser.add_argument("--filename", type=str, default=None, help="File name to save/load the model.")
        parser.add_argument("--trajectory_n", type=int, default=2500, help="Number of trajectories.")
        parser.add_argument("--iteration_n", type=int, default=1000, help="Number of iterations.")
        parser.add_argument("--gamma_q", type=float, default=0.5, help="Gamma quantile.")
        parser.add_argument("--laplace_f", type=float, default=0.0, help="Laplace smoothing factor.")
        parser.add_argument("--policy_f", type=float, default=1.0, help="Policy smoothing factor.")
        parser.add_argument("--max_trajectory_len", type=int, default=None, help="Maximum trajectory length.")
        parser.add_argument("--render", action="store_true", help="Render the environment during training and evaluation.") 
        parser.add_argument("--project", type=str, default="cross_entropy_agent_project", help="Name for the project for wandb")
        args = parser.parse_args()

        # Check if wandb is initialized (i.e., running a sweep)
        if wandb.run:
            # Override args with values from wandb.config
            args.env = wandb.config.env
            args.trajectory_n = wandb.config.trajectory_n
            args.iteration_n = wandb.config.iteration_n
            args.gamma_q = wandb.config.gamma_q
            args.laplace_f = wandb.config.laplace_f
            args.policy_f = wandb.config.policy_f
            args.max_trajectory_len = wandb.config.max_trajectory_len
            args.train = wandb.config.train if hasattr(wandb.config, 'train') else args.train
            if hasattr(wandb.config, 'filename'):
                args.filename = wandb.config.filename
            elif not args.filename:
                if (not os.path.exists("checkpoints")):
                    os.mkdir("checkpoints")
                args.filename = f"{checkpoints}/{wandb.run.name}.ckpt.npy"
            args.render = wandb.config.render if hasattr(wandb.config, 'render') else args.render


        env = gym.make(args.env)
        state_n = env.observation_space.n
        action_n = env.action_space.n

        agent = CrossEntropyAgent(state_n, action_n)

        if not args.train:
            agent.load(args.filename)

        trainer = Trainer(env, agent, render=args.render)

        if args.train:
            trainer.train(args.trajectory_n, args.iteration_n, 
                          args.gamma_q, args.max_trajectory_len or 2 * state_n,
                          args.laplace_f, args.policy_f)
            agent.save(args.filename)

        # Evaluate
        trainer.eval(args.trajectory_n, args.iteration_n, args.max_trajectory_len or 2 * state_n)

    # Check if the script is being run as part of a sweep
    if "WANDB_SWEEP_ID" in os.environ:
        wandb.init()
    else:
        # Initialize from CLI args
        wandb.init(project=args.project)
        
    main_function()
    wandb.finish()
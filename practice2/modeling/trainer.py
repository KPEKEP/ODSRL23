import torch
import time
import random
import logging
import wandb
from tqdm.auto import tqdm 
import os 
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    """Trains and evaluates the CrossEntropyAgent in a given environment."""

    def __init__(self, env, agent, gamma, device="cpu", dtype=torch.float32):
        self.env = env
        self.agent = agent.to(device)
        self.device = device
        self.dtype = dtype
        self.gamma = gamma

    def get_trajectories(self, n_trajectories, exploration, no_random):
        trajectories = []  # Final list of trajectories
        current_trajectories = [{'states': [], 
                                 'actions': [], 
                                 'action_extras':[], 
                                 'rewards': []} for _ in range(self.env.num_envs)]  # Ongoing trajectories for each environment
    
        states, infos = self.env.reset()
        total_rewards = [0] * self.env.num_envs
        trajectories_pbar = tqdm(total=n_trajectories, leave=False, position=1, desc="Trajectory generation")
        trajectories_count = 0 
        while len(trajectories) < n_trajectories:
            actions = []
            for i, state in enumerate(states):
                current_trajectories[i]['states'].append(state)
                
                action, extra = self.agent.get_action(state, exploration, no_random=no_random)
                actions.append(action)
                
                current_trajectories[i]['actions'].append(action)
                current_trajectories[i]['action_extras'].append(extra)
            
            obs, rewards, terminations, truncations, infos = self.env.step(actions)
    
            for i, (reward, termination, truncation) in enumerate(zip(rewards, terminations, truncations)):
                total_rewards[i] += reward
                current_trajectories[i]['rewards'].append(reward)
    
                # Check the stopping criteria
                if termination or truncation:
                    # Add the trajectory to the final list
                    trajectories.append(current_trajectories[i])
                    
                    # Reset for this environment
                    current_trajectories[i] = {'states': [], 'actions': [], 'action_extras': [], 'rewards': []}
                    total_rewards[i] = 0
                    
            states = obs
            trajectories_pbar.update(len(trajectories) - trajectories_count)
            trajectories_count = len(trajectories) 
        
        trajectories_pbar.update(n_trajectories)
        trajectories_pbar.close()
        print("\n")
        return trajectories[:n_trajectories]  # Return only up to n_trajectories

    def get_rewards(self, trajectories, discount_factor):
        return [self.get_total_reward(trajectory, discount_factor) for trajectory in trajectories]

    def train(self, trajectory_n, episode_n, 
              gamma_q, delta_q,
              exploration,
              goal, batch_size,
              fit_attempts,
              verbose=True, log=False):
        """Train the agent using the Cross Entropy method."""
        
        all_trajectories = []
        all_rewards = []
        
        for episode in tqdm(range(episode_n), desc="Train"):
            t = episode / episode_n
            max_traj = trajectory_n

            self.agent.eval()
            trajectories = self.get_trajectories(max_traj, exploration, False)           
            total_rewards = self.get_rewards(trajectories, self.gamma)

            total_rewards = torch.tensor(total_rewards, dtype=self.dtype)
            mean_total_reward = total_rewards.mean()
            min_total_reward = total_rewards.min()
            max_total_reward = total_rewards.max()
    
            # Log to wandb
            if log:
                wandb.log({
                    "Training/Mean_Total_Reward_disc": mean_total_reward.item(),
                    "Training/Min_Total_Reward_disc": min_total_reward.item(),
                    "Training/Max_Total_Reward_disc": max_total_reward.item()
                })
    
            if verbose:
                logger.info(f"Train {episode=} discounted {min_total_reward=} {mean_total_reward=} {max_total_reward=}")

            # Filtering the newly generated trajectories based on rewards
            quantile = total_rewards.quantile(gamma_q)
            elite_trajectories = [trajectory for i, trajectory in enumerate(trajectories) if total_rewards[i] >= quantile]
            elite_rewards = [reward for i, reward in enumerate(total_rewards) if total_rewards[i] >= quantile]

            # add the to the pool
            all_trajectories.extend(elite_trajectories)
            all_rewards.extend(elite_rewards)

            # filtering the pool
            if delta_q >= 0.0:
                quantile =  torch.tensor(all_rewards, dtype=self.dtype).quantile(delta_q)
                elite_trajectories = [trajectory for i, trajectory in enumerate(all_trajectories) if all_rewards[i] >= quantile]
                elite_rewards = [reward for i, reward in enumerate(all_rewards) if all_rewards[i] >= quantile]
    
                # store the elite pool to grow
                all_trajectories = elite_trajectories
                all_rewards = elite_rewards
            else:
                all_trajectories = []
                all_rewards = []
                
            logger.info(f"{len(elite_trajectories)=}")
            
            # Fit agent
            if elite_trajectories:
                elite_rewards_t = torch.tensor(elite_rewards, dtype=self.dtype)
                mean_elite_reward = elite_rewards_t.mean()
                min_elite_reward = elite_rewards_t.min()
                max_elite_reward = elite_rewards_t.max()
    
                if verbose:
                    logger.info(f"Elite stats {episode=} discounted {min_elite_reward=} {mean_elite_reward=} {max_elite_reward=}")

                fitness = tqdm(range(fit_attempts), desc="Fit")
                self.agent.train()
                for i in fitness:
                    loss = self.agent.fit(elite_trajectories, batch_size=batch_size)                
                    fitness.set_postfix({'loss': loss})
                print("\n")    
                logger.info(f"{episode=} {loss=}")
               
                reward_metrics = self.eval_episode(episode, max(1,int(trajectory_n * 0.25)), verbose=verbose, log=log)
                if reward_metrics[1] > goal:
                    logger.info("SOLVED!")
                    return
            else:
                logger.info("SKIPPED EPISODE")
        
    def eval_episode(self, episode, trajectory_n, verbose = True, log = False):
        self.agent.eval()
        trajectories = self.get_trajectories(trajectory_n, 0.0, True)            
        total_rewards = torch.tensor(self.get_rewards(trajectories, 1.0), dtype=self.dtype)
        
        
        mean_total_reward = total_rewards.mean()
        min_total_reward = total_rewards.min()
        max_total_reward = total_rewards.max()

        if log:
            # Log to wandb
            wandb.log({
                "Evaluation/Mean_Total_Reward_undisc": mean_total_reward.item(),
                "Evaluation/Min_Total_Reward_undisc": min_total_reward.item(),
                "Evaluation/Max_Total_Reward_undisc": max_total_reward.item()
            })

        if verbose:
            logger.info(f"Eval {episode=} undiscounted {mean_total_reward=} {min_total_reward=} {max_total_reward=}")
            
        return min_total_reward, mean_total_reward, max_total_reward

    def eval(self, trajectory_n, episode_n, 
             verbose=True, log=False):
        """Evaluate the agent's performance."""
        results = []
        for episode in tqdm(range(episode_n), desc="Eval"):
            min_total_reward, mean_total_reward, max_total_reward = self.eval_episode(episode, trajectory_n, verbose=verbose, log=log)
            
            results.append({
                'min_total_reward': min_total_reward,
                'mean_total_reward': mean_total_reward,
                'max_total_reward': max_total_reward
            })

        return results

    def get_total_reward(self, trajectory, discount_factor):
        """Calculate the total reward for a trajectory."""
        rewards_t = torch.tensor(trajectory['rewards'], dtype=self.dtype)
        discounts = torch.tensor([discount_factor**i for i in range(len(trajectory['rewards']))], dtype=self.dtype)
        return torch.sum(rewards_t * discounts)
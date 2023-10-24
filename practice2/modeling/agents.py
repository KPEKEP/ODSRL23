import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class BasicAgent(nn.Module):
    """Base class for an agent."""
    
    def __init__(self, device="cuda", dtype=torch.float32):
        super(BasicAgent, self).__init__()
        self.device = device
        self.dtype = dtype

    def save(self, filename):
        """Save the agent's data to a file."""
        torch.save(self.state_dict(), filename)

    def load(self, filename, env, device="cuda"):
        """Load the agent's data from a file."""
        self.load_state_dict(torch.load(filename))

    def get_action(self, state):
        """Get an action based on the current state."""
        raise Exception("Not implemented yet")

    def fit(self, elite_trajectories):
        """Train the agent using elite trajectories."""
        raise Exception("Not implemented yet")

    def check_compatibility(self, env):
        """Check if the agent is compatible with the given environment."""
        raise Exception("Not implemented yet")

class DeepCrossEntropyAgent(BasicAgent):
    """Agent that uses the Deep Cross Entropy method for policy optimization."""
    
    def __init__(self, env, depth=64, lr=0.01, device="cpu", dtype=torch.float32):                
        super().__init__(device=device, dtype=dtype)
        
        # Ensure environment compatibility
        assert self.check_compatibility(env), f"The agent is incompatible with the environment {env.single_observation_space=} {env.single_action_space=}"
        self.env = env
        
        # Determine the dimensionality of the state space
        if isinstance(env.single_observation_space, gym.spaces.Box):
            self.state_dim = torch.prod(torch.tensor(env.single_observation_space.shape, dtype=torch.int)).item()
        elif isinstance(env.single_observation_space, gym.spaces.Discrete):
            self.state_dim = env.single_observation_space.n
        else:
            raise Exception(f"The agent is incompatible with the observation space {env.single_observation_space=}")
        
        # Determine the number of possible actions
        if isinstance(env.single_action_space, gym.spaces.Box):
            self.action_n = torch.prod(torch.tensor(env.single_action_space.shape, dtype=torch.int)).item()
            action_space_v = env.single_action_space.high - env.single_action_space.low
            self.action_space_len = torch.tensor(action_space_v, dtype=self.dtype)
            self.action_space_mid = torch.tensor(env.single_action_space.low + action_space_v / 2.0, dtype=self.dtype)            
        elif isinstance(env.single_action_space, gym.spaces.Discrete):
            self.action_n = env.single_action_space.n
        else:
            raise Exception(f"The agent is incompatible with the action space {env.single_action_space=}")

        # Define the neural network architecture
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, depth, dtype=self.dtype),
            nn.ReLU(),
            nn.Linear(depth, self.action_n, dtype=self.dtype),            
        ).to(device)  # move the network to the chosen device
        
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Choose a loss function based on action space
        self.loss = nn.CrossEntropyLoss() if isinstance(env.single_action_space, gym.spaces.Discrete) else nn.MSELoss()

    def check_compatibility(self, env):
        """Checks compatibility with environment's observation and action space."""
        return isinstance(env.single_observation_space, (gym.spaces.Box, gym.spaces.Discrete)) and \
               isinstance(env.single_action_space, (gym.spaces.Box, gym.spaces.Discrete))

        
    def forward(self, x):
        """Forward pass of the neural network."""
        return self.net(x)
                
    def fit(self, elite_trajectories, batch_size=-1):
        """Training method using provided elite trajectories."""
        
        for trajectory in elite_trajectories:
            elite_states = trajectory['states']
            elite_actions = trajectory['actions']
            elite_action_extras = trajectory['action_extras']

            # Convert to tensors
            elite_states_tensor = torch.tensor(np.array(elite_states), dtype=self.dtype)
            if isinstance(self.env.single_action_space, gym.spaces.Discrete):
                elite_actions_tensor = torch.tensor(np.array(elite_actions), dtype=torch.long)
            else:
                elite_actions_tensor = torch.tensor(np.array(elite_action_extras), dtype=self.dtype)

            # Create TensorDataset and DataLoader for batching
            dataset = TensorDataset(elite_states_tensor, elite_actions_tensor)
            if batch_size > 0:
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            else:
                dataloader = DataLoader(dataset, batch_size=len(elite_states), shuffle=False)

            for batch_states, batch_actions in dataloader:
                batch_states = batch_states.to(self.device)
                batch_actions = batch_actions.to(self.device)

                self.optimizer.zero_grad()
                
                pred_actions = self.forward(batch_states)
                loss = self.loss(pred_actions, batch_actions)
                
                loss.backward()
                self.optimizer.step()
            
        return loss.item()

        
    def get_action(self, state, exploration, no_random = False):
        """Select an action based on the current state."""        
        state = torch.tensor(state, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            logits = self.forward(state)

            if isinstance(self.env.single_action_space, gym.spaces.Discrete):
                true_probs = torch.softmax(logits, dim=-1)
                uniform_probs = torch.ones_like(true_probs) / self.action_n
                action_probs = torch.lerp(true_probs, uniform_probs, exploration)
                if no_random:
                    action = torch.argmax(action_probs, dim=-1)
                else:
                    action = torch.multinomial(action_probs, 1)
                action = action.cpu().numpy().item()
                logits = logits.cpu()
            else:                
                true_action = logits.reshape(self.env.single_action_space.shape)
                noise = torch.normal(mean=self.action_space_mid, std=self.action_space_len)
                noise = noise * exploration
                action = true_action + noise
                action = action.cpu().numpy().tolist()
                logits = action

        return (action, logits)
                
class CrossEntropyAgent(BasicAgent):
    """Agent using the traditional Cross Entropy method for policy optimization."""
    
    def __init__(self, env, lr=1.0, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        # Ensure environment compatibility
        assert self.check_compatibility(env), "The agent is incompatible with the environment"
        
        # Determine the number of states and actions
        self.state_n = env.single_observation_space.n
        self.action_n = env.single_action_space.n
        self.actions = torch.arange(self.action_n, dtype=torch.long)
        
        # Initialize the model with uniform probabilities for each action
        self.model = torch.ones((self.state_n, self.action_n), dtype=self.dtype) / self.action_n
        self.lr = lr

    def check_compatibility(self, env):
        """Checks compatibility with discrete environment's observation and action space."""
        return isinstance(env.single_observation_space, gym.spaces.Discrete) and \
               isinstance(env.single_action_space, gym.spaces.Discrete)

    def get_action(self, state):
        """Sample an action based on the current policy for the given state."""
        action_probs = self.model[state]
        action = torch.multinomial(action_probs, 1).squeeze()
        return (action.item(), action_probs)

    def fit(self, trajectories):
        """Update the policy based on the provided trajectories."""
        new_model = torch.zeros_like(self.model, dtype=self.dtype)
        
        # Accumulate counts for state-action pairs
        for trajectory in trajectories:
            for s, a in zip(trajectory['states'], trajectory['actions']):
                new_model[s, a] += 1

        # Normalize the rows of new_model
        row_sums = new_model.sum(dim=1, keepdim=True)
        non_zero_rows = row_sums.squeeze() > 0
        new_model[non_zero_rows] /= row_sums[non_zero_rows].clamp(min=1)  # Avoid division by zero
        
        # Update the model using learning rate
        self.model = new_model*self.lr + self.model*(1-self.lr)
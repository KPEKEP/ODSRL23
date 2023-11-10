import matplotlib.pyplot as plt
import time
from Frozen_Lake import FrozenLakeEnv
from tqdm.auto import tqdm
import numpy as np

env = FrozenLakeEnv()

def init_v_values():
    v_values = {}
    for state in env.get_all_states():
        v_values[state] = 0
    return v_values

def get_q_values(v_values, gamma):
    q_values = {}
    for state in env.get_all_states():
        q_values[state] = {}
        for action in env.get_possible_actions(state):
            q_values[state][action] = 0
            for next_state in env.get_next_states(state, action):
                p = env.get_transition_prob(state, action, next_state)
                q_values[state][action] += p * (env.get_reward(state, action, next_state) + gamma * v_values[next_state])
    return q_values

def extract_policy(v_values, gamma):
    policy = {}
    q_values = get_q_values(v_values, gamma)
    for state in env.get_all_states():
        policy[state] = max(q_values[state], key=q_values[state].get) if q_values[state] else 0
    return policy

def value_iteration(gamma, iter_n):
    v_values = init_v_values()
    for i in tqdm(range(iter_n), desc="Value Iteration"):
        q_values = get_q_values(v_values, gamma)
        for state in env.get_all_states():
            v_values[state] = max(q_values[state].values()) if q_values[state] else 0
    return v_values

def policy_evaluation(policy, v_values, gamma):
    q_values = get_q_values(v_values, gamma)
    new_v_values = init_v_values()

    for state in env.get_all_states():
        action = policy[state]
        if action is not None:
            new_v_values[state] = q_values[state][action]

    return new_v_values

def policy_iteration(gamma, iter_n):
    v_values = init_v_values()

    # Initialize policy
    policy = {}
    for state in env.get_all_states():
        possible_actions = env.get_possible_actions(state)
        policy[state] = np.random.choice(possible_actions) if possible_actions else None

    for i in tqdm(range(iter_n), desc="Policy Iteration"):
        old_v_values = v_values.copy()
        v_values = policy_evaluation(policy, v_values, gamma)
        q_values = get_q_values(v_values, gamma)

        for state in env.get_all_states():
            policy[state] = max(q_values[state], key=q_values[state].get) if q_values[state] else None

        if old_v_values == v_values:
            break

    return policy, v_values



def test_policy(env, policy, num_episodes):
    total_rewards = 0
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        while True:
            action = policy[state]
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        total_rewards += episode_reward
    return total_rewards / num_episodes

gammas = [0.8, 0.9, 0.99]
iter_ns = [100, 500, 1000, 10000]

# Initialize the results dictionary
results = {'value_iteration': {}, 'policy_iteration': {}}

for iter_n in tqdm(iter_ns, desc="Iteration"):
    results['value_iteration'][iter_n] = {'gamma': [], 'time': [], 'reward': []}
    results['policy_iteration'][iter_n] = {'gamma': [], 'time': [], 'reward': []}

    for gamma in tqdm(gammas, desc="Gamma"):
        # Value Iteration
        start_time = time.time()
        v_values = value_iteration(gamma, iter_n)
        policy = extract_policy(v_values, gamma)
        elapsed_time = time.time() - start_time
        reward = test_policy(env, policy, 1000)
        results['value_iteration'][iter_n]['gamma'].append(gamma)
        results['value_iteration'][iter_n]['time'].append(elapsed_time)
        results['value_iteration'][iter_n]['reward'].append(reward)
        
        # Policy Iteration
        start_time = time.time()
        policy, _ = policy_iteration(gamma, iter_n)
        elapsed_time = time.time() - start_time
        reward = test_policy(env, policy, 1000)
        results['policy_iteration'][iter_n]['gamma'].append(gamma)
        results['policy_iteration'][iter_n]['time'].append(elapsed_time)
        results['policy_iteration'][iter_n]['reward'].append(reward)

print("Plotting")

for iter_n in iter_ns:
    plt.figure(figsize=(12, 5))
    plt.suptitle(f'Iteration: {iter_n}')

    plt.subplot(1, 2, 1)
    plt.plot(results['value_iteration'][iter_n]['gamma'], results['value_iteration'][iter_n]['time'], label='Value Iteration')
    plt.plot(results['policy_iteration'][iter_n]['gamma'], results['policy_iteration'][iter_n]['time'], label='Policy Iteration')
    plt.xlabel('Gamma')
    plt.ylabel('Time (seconds)')
    plt.title('Time Comparison')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(results['value_iteration'][iter_n]['gamma'], results['value_iteration'][iter_n]['reward'], label='Value Iteration')
    plt.plot(results['policy_iteration'][iter_n]['gamma'], results['policy_iteration'][iter_n]['reward'], label='Policy Iteration')
    plt.xlabel('Gamma')
    plt.ylabel('Average Reward')
    plt.title('Reward Comparison')
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
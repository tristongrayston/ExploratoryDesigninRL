import torch as T
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from PPOExample import Agent as PPO
import seaborn as sns


env = gym.make("LunarLander-v2", continuous=True)
device = T.device("cuda" if T.cuda.is_available() else "cpu")


episode_seed = np.random.randint(0, 100)
observation, info = env.reset(seed=episode_seed)

# --- HYPERPARAMETERS ---
EPOCHS = 25
ITERATIONS = 1000
TS_PER_ITER = 2000
POLICY_MAX_TRAIN = 80
CRITIC_MAX_TRAIN = 80

PPO_Agent = PPO(n_actions=4, c1=1.0, c2=0.01, input_dims=8, continuous=True)

# --- Bookkeeping ---
eps_rewards = []

# --- Training Loop ---
for episode in range(ITERATIONS):

    ep_reward = PPO_Agent.rollout(env, TS_PER_ITER)
    eps_rewards.append(ep_reward)

    print(ep_reward)

env.close()

# --- Evaluation Run ---

# --- Remake gym with correct render mode ---
env = gym.make("LunarLander-v2", continuous=True, render_mode="human")
observation, info = env.reset(seed=episode_seed)

# --- Set to evaluation mode ---
PPO_Agent.eval = True
PPO_Agent.actor.eval()

ep_reward = PPO_Agent.rollout(env, TS_PER_ITER)

print(eps_rewards)

plt.plot(eps_rewards, label="Episode Mean Rewards")
plt.legend()
plt.grid()
plt.title("Rewards Per Episode")
plt.xlabel("Episode")
plt.ylabel("Rewards")

plt.show()
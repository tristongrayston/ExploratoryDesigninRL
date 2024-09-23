import torch as T
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from PPOExample import Agent as PPO
import seaborn as sns


env = gym.make("LunarLander-v2")
device = T.device("cuda" if T.cuda.is_available() else "cpu")


episode_seed = np.random.randint(0, 100)
observation, info = env.reset(seed=episode_seed)

# --- HYPERPARAMETERS ---
EPOCHS = 25
ITERATIONS = 500
TS_PER_ITER = 2000
POLICY_MAX_TRAIN = 80
CRITIC_MAX_TRAIN = 80

PPO_Agent = PPO(n_actions=4, c1=1.0, c2=0.01, input_dims=8)

# --- Bookkeeping ---
eps_rewards = []

# --- Training Loop ---
for episode in range(ITERATIONS):

    train_data, ep_reward = PPO_Agent.rollout(env, TS_PER_ITER)
    eps_rewards.append(ep_reward)

    # Neat trick, thanks Eden Meyers! 
    permute_idx = np.random.permutation(len(train_data))

    # --- Batchify ---

    # This is a bit tedious, I also want to mess around with how memory bank works 
    # I'm not sure how much modifying the memory to make it more intelligent sampling wise
    # might impact performance. That being said, it would be good to test out. 

    obs = T.tensor(np.array(train_data[0])[permute_idx], dtype=T.float32, device=device)
    acts = T.tensor(np.array(train_data[1])[permute_idx], dtype=T.float32, device=device)
    rew = T.tensor(np.array(train_data[2])[permute_idx], dtype=T.float32, device=device)
    advantages = T.tensor(np.array(train_data[3])[permute_idx], dtype=T.float32, device=device)
    act_log_probs = T.tensor(np.array(train_data[4])[permute_idx], dtype=T.float32, device=device)    
    dones = T.tensor(np.array(train_data[5])[permute_idx], dtype=T.float32, device=device)    

    returns = PPO_Agent.discount_rewards(train_data[2])[permute_idx]

    # --- Train Policy ---
    PPO_Agent.train_actor(obs, acts, act_log_probs, advantages, POLICY_MAX_TRAIN)

    # --- Train Critic ---
    PPO_Agent.train_critic(obs, returns, CRITIC_MAX_TRAIN)

    print(ep_reward)

print(eps_rewards)

plt.plot(eps_rewards, label="Episode Mean Rewards")
plt.legend()
plt.grid()
plt.title("Rewards Per Episode")
plt.xlabel("Episode")
plt.ylabel("Rewards")

plt.show()
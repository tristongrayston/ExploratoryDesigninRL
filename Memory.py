
import os 
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import time
from collections import deque

device = T.device("cuda" if T.cuda.is_available() else "cpu")

'''
The way I'm understanding GAE to work is that you need multiple timesteps to get the best performance of GAE.
That means that randomly sampling actions and outcomes will provide a short end of the stick when it comes to implementation.
What we could do to circumnavigate this is implement a queue of batches in memory, and delete specifically those trajectories as needed.


'''

# obs, action, distributions, rewards, values, act_log_probs, dones

class PPOMemory:
    def __init__(self, batch_size, max_batches=300):
        # we keep memory in mind with lists
        self.tmp_storage = [[], [], [], [], [], [], []] # obs, action, distributions, rewards, values, act_log_probs, dones
        self.batch_size = batch_size
        
        #Batch queue
        self.batch_memory = []
        self.batch_memory_size = len(self.batch_memory)
        self.max_batches = max_batches

    def store_memory(self, obs, action, rewards, values, act_log_probs, dones):
        
        # --- Store values into buffer ---
        self.states = obs
        self.actions = action
        self.rewards = rewards
        self.logprobs = act_log_probs
        self.vals = values
        
        self.dones = dones

        # If we have enough memories to store a batch, then store a batch.
        #if len(self.states) >= self.batch_size:
        #    self._create_batch()

    def _clear_tmp_memory(self):
        self.tmp_storage = [[], [], [], [], [], [], []]

    def create_batch(self):
        ''' Returns a memory batch of size batch_size. '''

        # --- Replace vf with advantages ---

        # Neat trick, thanks Eden Meyers! 
        permute_idx = np.random.permutation(len(self.tmp_storage))

        obs = T.tensor(np.array(self.tmp_storage[0])[permute_idx], dtype=T.float32, device=device)
        acts = T.tensor(np.array(self.tmp_storage[1])[permute_idx], dtype=T.float32, device=device)
        #logits = T.tensor(np.array(self.tmp_storage[2])[permute_idx], dtype=T.float32, device=device)
        rew = T.tensor(np.array(self.tmp_storage[3])[permute_idx], dtype=T.float32, device=device)
        advantages = T.tensor(np.array(self.tmp_storage[4])[permute_idx], dtype=T.float32, device=device)
        act_log_probs = T.tensor(np.array(self.tmp_storage[5])[permute_idx], dtype=T.float32, device=device)    
        dones = T.tensor(np.array(self.tmp_storage[6])[permute_idx], dtype=T.float32, device=device)    

        # --- Check the size of our batch memory ---
        
        if self.batch_memory_size > self.max_batches:
            rnd_index = np.randint(0, self.batch_memory_size)
            del self.batch_memory[rnd_index]

        # --- Put a tuple of tensors in the batch memory ---
        self.batch_memory.append((obs, acts, rew, advantages, act_log_probs, dones))

        # --- Clear our temp memory ---
        self._clear_tmp_memory()

    def return_batch(self):
        rnd_index = np.random.randint(0, self.batch_memory_size)
        return self.batch_memory[rnd_index]
        



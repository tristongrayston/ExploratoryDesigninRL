
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

class PPOMemory:
    def __init__(self, batch_size, max_batches=300):
        # we keep memory in mind with lists
        self.states = []
        self.actions = []
        self.logprobs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        
        #Batch queue
        self.batch_memory = []
        self.batch_memory_size = len(self.batch_memory)
        self.max_batches = max_batches

    def store_memory(self, state, action, probs, vals, reward, done):
        
        # Store our vals into our queue
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

        # If we have enough memories to store a batch, then store a batch.
        if len(self.states) >= self.batch_size:
            self._create_batch()

    def _clear_tmp_memory(self):
        self.states = []
        self.logprobs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def _create_batch(self):
        ''' Returns a memory batch of size batch_size. '''

        # retrieves the batch_size states
        states_T = T.stack(self.states[:self.batch_size]).to(device)
        act_logprob_tens = T.tensor(self.logprobs[:self.batch_size]).to(device)
        vals_tens = T.tensor(self.vals[:self.batch_size], dtype=T.float32).to(device)
        act_tens = T.tensor(self.actions[:self.batch_size]).to(device)
        rew_tens = T.tensor(self.rewards[:self.batch_size]).to(device)
        dones_tens = T.tensor(self.dones[:self.batch_size]).to(device)

        # Check the size of our batch memory
        
        if self.batch_memory_size > self.max_batches:
            rnd_index = np.randint(0, self.batch_memory_size)
            del self.batch_memory[rnd_index]

        # Put a tuple of tensors in the batch memory
        self.batch_memory.append((states_T, act_logprob_tens, vals_tens, act_tens, rew_tens, dones_tens))

        # clear our temp memory

        self._clear_tmp_memory()

    def return_batch(self):
        rnd_index = np.randint(0, self.batch_memory_size)
        return self.batch_memory[rnd_index]
        



'''
Future Design tweaks:

1. Dropout layer, does it help?
2. Otherwise, some kind of normalization as data goes through the layers. 
3. A more intelligent method of sampling through memories (right now, training on the most recent episode)
4. Outputs right now go through a tanh activation to get it within the range of (-1, 1). This likely is a shitty solution.
5. No clue if a synchronized model with diverging output layers is better than two distinct models. More work needs to be done here to figure that out.

6. This model is for a continuous output space, so our outputs are mean vals with constant variance. Likely, this is a mistake.

'''

import os 
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions import Normal
import time
from Memory import PPOMemory

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class Agent(nn.Module):
        # An interesting note - implementations exist where actor and critic share 
        # the same NN, differentiated by a singular layer at the end. 
        # food for thought.
    
    def __init__(self, n_actions, c1, c2, input_dims, gamma=0.99, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, buffer_size=32, n_epochs=10, 
                 target_kl_div = 0.01, act_min_val = -1, act_max_val = 1,
                 actor_LR=1e-4, crit_LR=1e-4, annealing=True, continuous=False):
        
        super(Agent, self).__init__()

        #           --- Hyperparams ---
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.gae_lambda = gae_lambda
        self.c1 = c1
        self.c2 = c2
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_actions = n_actions
        self.min_val = act_min_val
        self.max_val = act_max_val

        #           --- Actor Critic ---
        self.actor = self._create_model(input_dims, n_actions, actor=True).float().to(device)
        self.optimizer_actor = T.optim.Adam(self.actor.parameters(), actor_LR)

        self.critic = self._create_model(input_dims, 1, actor=False).float().to(device)
        self.optimizer_critic = T.optim.Adam(self.critic.parameters(), crit_LR)

        #           --- Memory ---
        self.memory = PPOMemory(self.batch_size)

        #           --- Misc ---
        self.target_kld = target_kl_div
        self.criterion = nn.MSELoss()
        self.continuous = continuous
        if continuous == True:
            self.variance = 0.3*T.ones(n_actions) # 0.3 is the current variance. We should probably change this.

        self.annealing = annealing
        if annealing == True:
            self.anneal_lr_actor = T.optim.lr_scheduler.StepLR(self.optimizer_actor, buffer_size*5, gamma=0.3)
            self.anneal_lr_critic = T.optim.lr_scheduler.StepLR(self.optimizer_critic, buffer_size*5, gamma=0.3)

        self.training_steps = 0
    

    def _create_model(self, input_dims, output_dims, actor):
        ''' private function meant to create the same model with varying input/output dims. '''
        if actor == True:
            model = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, 64),
            nn.Tanh(),

            nn.Linear(64, output_dims),
            nn.Tanh()
            )
            return model
        model = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, output_dims)
        )
        return model

    def get_vf(self, x):
        ''' retrieve the value function for that state as determined by critic. '''
        return self.critic.forward(x.to(device))
    
    def get_gae_and_returns(self, rewards, vf_t, reward_std, reward_mean, dones):
        ''' As seen here: https://arxiv.org/pdf/1506.02438.pdf
            An estimation for the advantage function. 
            GAE = r_t - gamma*lambda*vf_(t+1) + vf(t).

            We create the advantage functions for an agent given a batch. 
        '''
        # get value functions for next state
        vf_t1_tens = T.squeeze(T.stack(self.memory.vals[1:self.batch_size+1])).to(device)

        # vf's of terminal states are always 0
        not_dones = T.logical_not(dones, out=T.empty(self.batch_size)).to(device)
        #print(not_dones.shape)
        vf_t1_tens = vf_t1_tens*not_dones

        # get our GAE function
        gae = rewards + self.gamma*self.gae_lambda*vf_t1_tens - vf_t

        # sum over the rewards to get returns

        # replace last reward with value function for that state
        # to approx all rewards for future 
        rewards = rewards[:-1]
        rewards = T.cat([rewards, vf_t1_tens[-1].unsqueeze(0)]).to(device)

        returns = T.ones(rewards.shape).to(device)
        
        # apply discounting
        for i in range(self.batch_size):
            return_t = returns[i:]
            gammas = self.gamma + T.zeros(return_t.shape).to(device)
            exp = T.arange(0, self.batch_size-i).to(device)
            gammas = T.pow(gammas, exp).to(device)
            return_t = T.sum(return_t*gammas).to(device)
            returns[i] *= return_t

        # scale
        returns = (returns - reward_mean) / (reward_std)
        returns = returns.float()
        
        return gae.clone().detach(), returns.clone().detach()
    
    def discount_rewards(self, rewards, gamma=0.99):
        """
        Return discounted rewards based on the given rewards and gamma param.

        Credit: Eden Meyer
        """
        new_rewards = [float(rewards[-1])]
        for i in reversed(range(len(rewards)-1)):
            new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
        return T.tensor(new_rewards[::-1], dtype=T.float32, device=device)

    def calculate_gaes(self, rewards, values, gamma=0.99, decay=0.97):
        """
        Return the General Advantage Estimates from the given rewards and values.
        Paper: https://arxiv.org/pdf/1506.02438.pdf
        Credit: Eden Meyer
        """
        # This is only important if you're running on GPU
        values = T.stack(values).detach().cpu().numpy()

        next_values = np.concatenate([values[1:], [[0]]])
        deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

        gaes = [deltas[-1]]
        for i in reversed(range(len(deltas)-1)):
            gaes.append(deltas[i] + decay * gamma * gaes[-1])

        return np.array(gaes[::-1])
    
    def get_action_and_vf(self, x):
        ''' get distribution over actions and associated vf '''

        # --- Discrete Actions ---
        if self.continuous == False:
            logits = self.actor(x)
            probs = Categorical(logits=logits)
            action = probs.sample()
            return action, probs, probs.entropy(), self.critic.forward(x)
        
        # --- Continuous Actions ---
        else:
            means = self.actor(x)
            #print(means.tolist())
            #print(type(means.tolist()))
            distributions = Normal(means, self.variance) 
            print(dist)
            samples = distributions.sample(4)
            samples_clamped = samples.clamp(self.min_val, self.max_val) 

            return samples_clamped, samples, 0, self.critic.forward(x)


    def rollout(self, env, max_steps):
        """
        Takes the environment and performs one episode of the environment. 

        Noteworthy: the implementation here (I think) will turn training data into batches per episode. Maybe that is 
        the way to go? I'm thinking bootstrapping after some 16th step might be better. Not quite sure.
        """
        train_data = [[], [], [], [], [], []] # obs, action, rewards, values, act_log_probs, dones
        obs, _ = env.reset()

        ep_reward = 0.0

        # --- Perform Rollout ---
        for _ in range(max_steps):
            action, logits, entropy, vals = self.get_action_and_vf(T.tensor(obs, dtype=T.float32, device=device))
            log_prob = logits.log_prob(action).item()

            next_obs, reward, done, trun, _ = env.step(action.item())
            for i, item in enumerate((obs, action.item(), reward, vals, log_prob, done)):
                train_data[i].append(item)

            obs = next_obs
            ep_reward += reward 
            if done:
                break
        
        # --- Get GAE, replacing values with advantages. --- 
        
        train_data[3] = self.calculate_gaes(train_data[2], train_data[3])
        return train_data, ep_reward

    def train_actor(self, obs, actions,
                    act_log_probs, adv_tensor, max_train_steps):
        '''
        Function that trains specifically the policy network. Admittedly, might need to make some changes to the model
        as we have no shared layers.
        '''
        for train_step in range(max_train_steps):
            # Forget gradients from last learning step
            self.optimizer_actor.zero_grad()

            # --- Get Ratio ---

            logits = self.actor(obs)
            logits = Categorical(logits)
            log_probs = logits.log_prob(actions) 
            prob_ratios = T.exp(log_probs - act_log_probs).to(device)

            # --- Clip Loss ---
            clip_loss = prob_ratios.clamp(1 - self.policy_clip, 1 + self.policy_clip) 

            # --- Full Policy Loss --- 
            policy_loss = -(T.min(prob_ratios*adv_tensor, clip_loss*adv_tensor)).mean()
            policy_loss.backward()

            self.optimizer_actor.step()

            kld = (act_log_probs - log_probs).mean()
            if kld >= self.target_kld:
                break

    def train_critic(self, obs, returns, max_train_steps):
        '''
        Function that trains specifically the critic network. Admittedly, might need to make some changes to the model
        as we have no shared layers.
        '''
        for train_step in range(max_train_steps):
            # --- Zero out Grad ---
            self.optimizer_critic.zero_grad()

            # --- L2 Loss between Critic and returns ---

            val = self.critic(obs)
            loss = (val - returns)**2
            loss = loss.mean()

            # --- Backwards --- 
            loss.backward()

            self.optimizer_critic.step()

    def learn(self):
        '''
        This function is responsible for the backpropogation and learning of our agent. 
        It learns on minibatches and is called until the entire buffer is empty from the minibatch called. 
        '''

        # Certain things need to be done relative to the entire buffer, such as reward scaling.
        reward_std = np.array(self.memory.rewards).std()
        reward_mean = np.array(self.memory.rewards).mean()
        
        for _ in range(self.buffer_size):

            # Forget gradients from last learning step
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()

            # retrieve memories from last batch
            state_tens, act_logprob_tens, vals_tens, act_tens, rew_tens, dones_tens = self.memory.get_memory_batch()

            # clip rewards
            rew_tens = T.clamp(rew_tens, -1.0, 1.0)

            # Retrieve advantages and returns for this minibatch
            adv_tensor, returns = self.get_gae_and_returns(rew_tens, vals_tens, reward_std, reward_mean, dones_tens)
            
            #           --- Actor and Entropy Loss ---

            # Send our state through our actors
            logits=self.actor(state_tens).to(device)
            #print(logits)
            new_probs = Categorical(logits)

            # get prob of our action
            prob_of_action = new_probs.log_prob(act_tens)

            # Entropy Loss
            entropy_loss = T.mean(new_probs.entropy()).to(device)

            # Get probability raio
            prob_ratios = T.exp(prob_of_action - act_logprob_tens).to(device)

            maximum_ratio = T.max(prob_ratios).detach().numpy()

            # Clip Max Tensor
            clip_max = T.tensor(1+self.policy_clip, dtype=T.float32).expand(self.batch_size, 1).to(device)

            # Clip Min Tensor
            clip_min =  T.tensor(1-self.policy_clip, dtype=T.float32).expand(self.batch_size, 1).to(device)


            # policy loss
            policy_loss = (T.min((prob_ratios), T.clamp(prob_ratios, clip_min, clip_max))*adv_tensor).to(device)
            policy_loss = T.mean(policy_loss).to(device)

            #           --- Critic Loss ---

            approx_val = T.flatten(self.critic.forward(state_tens))
            
            crit_loss = self.criterion(approx_val, returns)

            # Implementation detail in 'Implementation Matters in Deep Policy Gradients'
            crit_loss_clip = T.clamp(crit_loss, 1-self.policy_clip, 1+self.policy_clip)
            crit_loss = T.min(crit_loss, crit_loss_clip)
            crit_loss = crit_loss.float()

            #           --- Total Loss ---

            # Implementation detail in 'Implementation Matters in Deep Policy Gradients'
            # No entropy loss
            loss = -policy_loss + self.c1*crit_loss # - self.c2*entropy_loss

            #print('policy_loss: ', policy_loss)
            #print('crit loss: ', crit_loss)
            #print('entropy loss: ', entropy_loss)
            #print(loss)

            # backward pass
            loss.backward()

            self.optimizer_actor.step()
            self.optimizer_critic.step()

            if self.annealing == True:
                self.anneal_lr_actor.step()
                self.anneal_lr_critic.step()
                #print("### Learning Rate : ", self.anneal_lr_actor.get_last_lr() , " ###")
                #self.training_steps += 1

            # Exponential Decay on C2
            self.c2 *= 0.999


        #print(self.memory.vals.shape)

        return policy_loss.detach(), crit_loss.detach(), loss.detach(), maximum_ratio

        


##A
    
    

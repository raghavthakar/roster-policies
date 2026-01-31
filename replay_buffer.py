import numpy as np
import torch
import random

class MultiAgentReplayBuffer:
    def __init__(self, max_size, agent_ids, obs_dims, act_dims, device):
        self.max_size = max_size
        self.agent_ids = agent_ids  # List of agent keys e.g. ['agent_0', 'agent_1']
        self.ptr = 0
        self.size = 0
        self.device = device

        # Initialize storage for each agent
        self.obs = {agent: np.zeros((max_size, obs_dims[agent])) for agent in agent_ids}
        self.next_obs = {agent: np.zeros((max_size, obs_dims[agent])) for agent in agent_ids}
        self.actions = {agent: np.zeros((max_size, act_dims[agent])) for agent in agent_ids}
        
        # Rewards and dones are usually shared or per-agent, but typically we store per-agent 
        # to be safe, even if they are identical in cooperative tasks.
        self.rewards = {agent: np.zeros((max_size, 1)) for agent in agent_ids}
        self.dones = {agent: np.zeros((max_size, 1)) for agent in agent_ids}

    def add(self, obs_dict, action_dict, reward_dict, next_obs_dict, done_dict):
        """
        Add a transition. Expects dictionaries with agent_id keys.
        """
        for agent in self.agent_ids:
            self.obs[agent][self.ptr] = obs_dict[agent]
            self.next_obs[agent][self.ptr] = next_obs_dict[agent]
            self.actions[agent][self.ptr] = action_dict[agent]
            self.rewards[agent][self.ptr] = reward_dict[agent]
            self.dones[agent][self.ptr] = done_dict[agent]

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        batch_obs = {}
        batch_next_obs = {}
        batch_actions = {}
        batch_rewards = {}
        batch_dones = {}

        for agent in self.agent_ids:
            batch_obs[agent] = torch.FloatTensor(self.obs[agent][ind]).to(self.device)
            batch_next_obs[agent] = torch.FloatTensor(self.next_obs[agent][ind]).to(self.device)
            batch_actions[agent] = torch.FloatTensor(self.actions[agent][ind]).to(self.device)
            batch_rewards[agent] = torch.FloatTensor(self.rewards[agent][ind]).to(self.device)
            batch_dones[agent] = torch.FloatTensor(self.dones[agent][ind]).to(self.device)

        return batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones
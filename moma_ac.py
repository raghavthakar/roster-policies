import torch
import torch.nn.functional as F
import numpy as np
import copy

from actor import MultiHeadActor
from critics import CentralizedCritic
from replay_buffer import MultiAgentReplayBuffer

class MultiAgentTD3:
    def __init__(self, 
                 agent_ids, 
                 obs_dims, 
                 act_dims, 
                 max_action=1.0, 
                 device="cpu",
                 lr_a=3e-4, 
                 lr_c=3e-4, 
                 gamma=0.99, 
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2):
        
        self.agent_ids = agent_ids
        self.device = device
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

        # Calculate dimensions for Centralized Critic
        # Flatten joint state and joint action
        joint_state_dim = sum(obs_dims.values())
        joint_action_dim = sum(act_dims.values())
        max_obs_dim = max(obs_dims.values())

        # --- Actors ---
        self.actor = MultiHeadActor(agent_ids, obs_dims, act_dims, max_obs_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a)

        # --- Critics ---
        self.critic = CentralizedCritic(joint_state_dim, joint_action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

        # --- Replay Buffer ---
        # Note: Initialized externally or here. We'll init here for simplicity if max_size provided
        # but the class signature doesn't ask for max_size. 
        # For this example, we assume buffer is passed to train()

    def select_action(self, obs_dict, noise_scale=0.1):
        """Select actions for all agents at current step"""
        actions = {}
        for agent in self.agent_ids:
            obs_tensor = torch.FloatTensor(obs_dict[agent]).unsqueeze(0).to(self.device)
            action = self.actor.get_action_single(agent, obs_tensor)
            
            # Add exploration noise
            if noise_scale > 0:
                noise = np.random.normal(0, noise_scale, size=action.shape)
                action = np.clip(action + noise, -self.max_action, self.max_action)
            
            actions[agent] = action
        return actions

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # 1. Sample Replay Buffer
        obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size)

        # Prepare joint tensors (Concatenate all agents)
        # We need a consistent order, so we rely on self.agent_ids list
        with torch.no_grad():
            # Generate Target Actions with Noise (Target Policy Smoothing)
            next_actions_dict = self.actor_target(next_obs)
            
            joint_next_action_list = []
            joint_reward = 0  # Assuming cooperative: reward is same for all, or we sum?
            # In MaMuJoCo, rewards are usually identical for all agents. 
            # Even if they are identical, summing will make no difference
            # But why inflate the values? Let's use only the first agent's reward (they are identical)
            joint_reward = joint_reward = rewards[self.agent_ids[0]]
            joint_done = dones[self.agent_ids[0]]

            for agent in self.agent_ids:
                noise = (torch.randn_like(next_actions_dict[agent]) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (next_actions_dict[agent] + noise).clamp(-self.max_action, self.max_action)
                joint_next_action_list.append(next_action)
            
            joint_next_action = torch.cat(joint_next_action_list, dim=1)
            joint_next_obs = torch.cat([next_obs[a] for a in self.agent_ids], dim=1)

            # Target Q-values
            target_Q1, target_Q2 = self.critic_target(joint_next_obs, joint_next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = joint_reward + (1 - joint_done) * self.gamma * target_Q

        # 2. Critic Update
        joint_state = torch.cat([obs[a] for a in self.agent_ids], dim=1)
        joint_action = torch.cat([actions[a] for a in self.agent_ids], dim=1)

        current_Q1, current_Q2 = self.critic(joint_state, joint_action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 3. Actor Update (Delayed)
        if self.total_it % self.policy_freq == 0:
            
            # We need the current policy's actions to calculate the gradient
            # But we must be careful:
            # We want to differentiate wrt the actions output by the actor.
            current_actions_dict = self.actor(obs)
            current_joint_action_list = []
            
            for agent in self.agent_ids:
                current_joint_action_list.append(current_actions_dict[agent])
            
            current_joint_action = torch.cat(current_joint_action_list, dim=1)

            # Maximize Q1
            actor_loss = -self.critic.Q1(joint_state, current_joint_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # 4. Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
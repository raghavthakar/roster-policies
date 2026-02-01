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
                 preference_dim,  # FIX: Added preference_dim
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
        self.preference_dim = preference_dim

        # Calculate dimensions for Centralized Critic
        # Flatten joint state and joint action
        joint_state_dim = sum(obs_dims.values())
        joint_action_dim = sum(act_dims.values())
        max_obs_dim = max(obs_dims.values())

        # --- Actors ---
        # Pass preference_dim to Actor
        self.actor = MultiHeadActor(agent_ids, obs_dims, act_dims, max_obs_dim, preference_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a)

        # --- Critics ---
        # Pass preference_dim to Critic (Output is now vector)
        self.critic = CentralizedCritic(joint_state_dim, joint_action_dim, preference_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

    def select_action(self, obs_dict, preference, noise_scale=0.1):
        """
        Select actions for all agents at current step.
        FIX: Takes preference vector as input.
        """
        actions = {}
        for agent in self.agent_ids:
            # Prepare inputs
            obs_tensor = torch.FloatTensor(obs_dict[agent]).unsqueeze(0).to(self.device)
            pref_tensor = torch.FloatTensor(preference).unsqueeze(0).to(self.device)
            
            # Pass preference to actor
            action = self.actor.get_action_single(agent, obs_tensor, pref_tensor)
            
            # Add exploration noise
            if noise_scale > 0:
                noise = np.random.normal(0, noise_scale, size=action.shape)
                action = np.clip(action + noise, -self.max_action, self.max_action)
            
            actions[agent] = action
        return actions

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # 1. Sample Replay Buffer (Now includes preferences)
        obs, actions, rewards, next_obs, dones, preferences = replay_buffer.sample(batch_size)

        # Prepare joint tensors (Concatenate all agents)
        with torch.no_grad():
            # Generate Target Actions with Noise (Target Policy Smoothing)
            # FIX: Pass preferences to target actor
            next_actions_dict = self.actor_target(next_obs, preferences)
            
            joint_next_action_list = []
            
            # In MOMA-AC, agents share the vector reward. We take agent_0's reward as the team reward.
            # rewards shape: [batch, preference_dim]
            joint_reward = rewards[self.agent_ids[0]] 
            joint_done = dones[self.agent_ids[0]]

            for agent in self.agent_ids:
                noise = (torch.randn_like(next_actions_dict[agent]) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (next_actions_dict[agent] + noise).clamp(-self.max_action, self.max_action)
                joint_next_action_list.append(next_action)
            
            joint_next_action = torch.cat(joint_next_action_list, dim=1)
            joint_next_obs = torch.cat([next_obs[a] for a in self.agent_ids], dim=1)

            # --- MOMA-TD3 Logic for Target Q (Eq. 4, 5, 6) ---
            # 1. Get Vector Q-values from both targets
            # Output shape: [batch, preference_dim]
            target_Q1_vec, target_Q2_vec = self.critic_target(joint_next_obs, joint_next_action, preferences)
            
            # 2. Scalarize using the batch preferences (Dot product)
            # (batch, dim) * (batch, dim) -> sum -> (batch, 1)
            scalar_Q1 = (target_Q1_vec * preferences).sum(dim=1, keepdim=True)
            scalar_Q2 = (target_Q2_vec * preferences).sum(dim=1, keepdim=True)
            
            # 3. Find index of the critic with lower SCALARIZED value
            # [cite_start]This implements the "min-of-two" in utility space to prevent overestimation [cite: 14]
            # Create a mask where Q1 < Q2
            mask = (scalar_Q1 < scalar_Q2).float() # 1 if Q1 is smaller, 0 otherwise
            
            # 4. Select the VECTOR corresponding to the lower scalar
            target_Q_vec = mask * target_Q1_vec + (1 - mask) * target_Q2_vec
            
            # 5. Bellman Update on the VECTOR
            target_Q = joint_reward + (1 - joint_done) * self.gamma * target_Q_vec

        # 2. Critic Update (Eq. 9)
        joint_state = torch.cat([obs[a] for a in self.agent_ids], dim=1)
        joint_action = torch.cat([actions[a] for a in self.agent_ids], dim=1)

        # Get current vector estimates
        current_Q1, current_Q2 = self.critic(joint_state, joint_action, preferences)

        # Loss is sum of MSE over vector components
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 3. Actor Update (Delayed)
        if self.total_it % self.policy_freq == 0:
            
            # Get current policy actions
            current_actions_dict = self.actor(obs, preferences)
            current_joint_action_list = []
            
            for agent in self.agent_ids:
                current_joint_action_list.append(current_actions_dict[agent])
            
            current_joint_action = torch.cat(current_joint_action_list, dim=1)

            # --- MOMA-TD3 Logic for Actor Loss (Eq. 10, 11, 12) ---
            # 1. Get Q1 Vector
            q1_vec = self.critic.Q1(joint_state, current_joint_action, preferences)
            
            # 2. Scalarize using preference vector
            utility = (q1_vec * preferences).sum(dim=1)
            
            # 3. Maximize Utility (Minimize negative utility)
            actor_loss = -utility.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            # 4. Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
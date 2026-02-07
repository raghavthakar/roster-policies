import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadActor(nn.Module):
    def __init__(self, agent_ids, obs_dims, act_dims, max_obs_dim, preference_dim, hidden_dim=256):
        super(MultiHeadActor, self).__init__()
        self.agent_ids = agent_ids
        self.max_obs_dim = max_obs_dim
        self.preference_dim = preference_dim
        
        # Shared Trunk (Feature Extractor)
        # Input is: [Padded_Obs, Preference]
        self.trunk_l1 = nn.Linear(max_obs_dim + preference_dim, hidden_dim)
        self.trunk_l2 = nn.Linear(hidden_dim, hidden_dim)

        # Independent Heads for each agent
        self.heads = nn.ModuleDict()
        for agent in agent_ids:
            self.heads[agent] = nn.Linear(hidden_dim, act_dims[agent])

    def forward(self, obs_dict, preference):
        """
        Forward pass for ALL agents.
        Returns a dictionary of actions: {agent_id: action_tensor}
        
        preference: Tensor of shape [batch_size, preference_dim]
        """
        actions = {}
        
        for agent in self.agent_ids:
            x = obs_dict[agent]
            
            # 1. Zero-pad observation if strictly smaller than max_obs_dim
            if x.shape[1] < self.max_obs_dim:
                padding = torch.zeros(x.shape[0], self.max_obs_dim - x.shape[1]).to(x.device)
                x = torch.cat([x, padding], dim=1)
            
            # FIX: Concatenate preference w to the observation
            # x shape: [batch, max_obs_dim], preference shape: [batch, pref_dim]
            x = torch.cat([x, preference], dim=1)

            # 2. Pass through Shared Trunk
            x = F.relu(self.trunk_l1(x))
            x = F.relu(self.trunk_l2(x))
            
            # 3. Pass through Specific Head
            action = torch.tanh(self.heads[agent](x))
            actions[agent] = action
            
        return actions

    def get_action_single(self, agent_id, obs, preference):
        """
        Helper for inference/acting in environment (single agent, single obs)
        preference: Numpy array or Tensor [preference_dim]
        """
        # 1. Convert Obs to Tensor
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(next(self.parameters()).device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0) # [1, obs_dim]

        # 2. Convert Preference to Tensor
        if not isinstance(preference, torch.Tensor):
            preference = torch.FloatTensor(preference).to(obs.device)
        if preference.dim() == 1:
            preference = preference.unsqueeze(0) # [1, pref_dim]

        # 3. Pad Observation
        if obs.shape[1] < self.max_obs_dim:
            padding = torch.zeros(obs.shape[0], self.max_obs_dim - obs.shape[1]).to(obs.device)
            obs = torch.cat([obs, padding], dim=1)

        # Concatenate Preference
        x = torch.cat([obs, preference], dim=1)

        # Trunk
        x = F.relu(self.trunk_l1(x))
        x = F.relu(self.trunk_l2(x))
        
        # Head
        action = torch.tanh(self.heads[agent_id](x))
        
        return action.detach().cpu().numpy()[0]


class MomaMoEActor(nn.Module):
    def __init__(self, agent_ids, obs_dims, act_dims, max_obs_dim, preference_dim, hidden_dim=256, num_experts=3):
        super(MomaMoEActor, self).__init__()
        self.agent_ids = agent_ids
        self.max_obs_dim = max_obs_dim
        self.preference_dim = preference_dim
        self.num_experts = num_experts
        
        # Shared Trunk (Feature Extractor)
        # Input is: [Padded_Obs, Preference]
        self.trunk_l1 = nn.Linear(max_obs_dim + preference_dim, hidden_dim)
        self.trunk_l2 = nn.Linear(hidden_dim, hidden_dim)

        # Mixture of Experts Components
        self.experts = nn.ModuleDict()
        self.routers = nn.ModuleDict()

        for agent in agent_ids:
            # Experts: A list of linear layers for each agent
            self.experts[agent] = nn.ModuleList([
                nn.Linear(hidden_dim, act_dims[agent]) for _ in range(num_experts)
            ])
            # Router: Maps preference vector to expert weights
            self.routers[agent] = nn.Sequential(
                nn.Linear(preference_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_experts),
                nn.Softmax(dim=-1)
            )

    def forward(self, obs_dict, preference):
        """
        Forward pass for ALL agents using Mixture of Experts.
        Returns:
            actions: dictionary {agent_id: action_tensor}
            all_router_weights: dictionary {agent_id: weights_tensor}
        """
        actions = {}
        all_router_weights = {}

        for agent in self.agent_ids:
            x = obs_dict[agent]
            
            # 1. Zero-pad observation if strictly smaller than max_obs_dim
            if x.shape[1] < self.max_obs_dim:
                padding = torch.zeros(x.shape[0], self.max_obs_dim - x.shape[1]).to(x.device)
                x = torch.cat([x, padding], dim=1)
            
            # Concatenate preference w to the observation
            x = torch.cat([x, preference], dim=1)

            # 2. Pass through Shared Trunk
            x = F.relu(self.trunk_l1(x))
            feat = F.relu(self.trunk_l2(x))
            
            # 3. Router: Calculate weights for each expert based on preference
            weights = self.routers[agent](preference) # Shape: [batch, num_experts]
            all_router_weights[agent] = weights

            # 4. Experts: Calculate output from each expert
            # Stack outputs: [batch, num_experts, act_dim]
            expert_outs = torch.stack([exp(feat) for exp in self.experts[agent]], dim=1)
            
            # 5. Weighted Sum: Combine expert outputs using router weights
            # weights.unsqueeze(2) shape: [batch, num_experts, 1]
            # Multiply and sum over expert dimension (dim 1)
            action = torch.sum(weights.unsqueeze(2) * torch.tanh(expert_outs), dim=1)
            
            actions[agent] = action
            
        return actions, all_router_weights

    def get_action_single(self, agent_id, obs, preference):
        """
        Helper for inference/acting in environment (single agent, single obs)
        """
        # 1. Convert Obs to Tensor
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(next(self.parameters()).device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0) # [1, obs_dim]

        # 2. Convert Preference to Tensor
        if not isinstance(preference, torch.Tensor):
            preference = torch.FloatTensor(preference).to(obs.device)
        if preference.dim() == 1:
            preference = preference.unsqueeze(0) # [1, pref_dim]

        # 3. Pad Observation
        if obs.shape[1] < self.max_obs_dim:
            padding = torch.zeros(obs.shape[0], self.max_obs_dim - obs.shape[1]).to(obs.device)
            obs = torch.cat([obs, padding], dim=1)

        # Concatenate Preference
        x = torch.cat([obs, preference], dim=1)

        # Trunk
        x = F.relu(self.trunk_l1(x))
        feat = F.relu(self.trunk_l2(x))
        
        # Router
        weights = self.routers[agent_id](preference)

        # Experts
        expert_outs = torch.stack([exp(feat) for exp in self.experts[agent_id]], dim=1)
        
        # Weighted Sum
        action = torch.sum(weights.unsqueeze(2) * torch.tanh(expert_outs), dim=1)
        
        return action.detach().cpu().numpy()[0]
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
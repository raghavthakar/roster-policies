import torch
import torch.nn as nn
import torch.nn.functional as F

class CentralizedCritic(nn.Module):
    def __init__(self, joint_state_dim, joint_action_dim, preference_dim, hidden_dim=256):
        super(CentralizedCritic, self).__init__()
        
        input_dim = joint_state_dim + joint_action_dim + preference_dim
        
        # Critic 1
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, preference_dim)

        # Critic 2 (Twin)
        self.l4 = nn.Linear(input_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, preference_dim)

    def forward(self, joint_state, joint_action, preference):
        """
        Returns vector Q-values for both critics.
        Input: 
            joint_state: [batch, state_dim]
            joint_action: [batch, action_dim]
            preference: [batch, preference_dim]
        Output:
            q1: [batch, preference_dim]
            q2: [batch, preference_dim]
        """
        # Concatenate preference w to the input
        xu = torch.cat([joint_state, joint_action, preference], dim=1)
        
        # Critic 1 Forward
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        # Critic 2 Forward
        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)

        return x1, x2

    def Q1(self, joint_state, joint_action, preference):
        """Returns only the first vector Q-value for actor updates."""
        # Concatenate preference w
        xu = torch.cat([joint_state, joint_action, preference], dim=1)
        
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        
        return x1
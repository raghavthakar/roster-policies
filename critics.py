import torch
import torch.nn as nn
import torch.nn.functional as F

class CentralizedCritic(nn.Module):
    def __init__(self, joint_state_dim, joint_action_dim, hidden_dim=256):
        super(CentralizedCritic, self).__init__()
        
        # Critic 1
        self.l1 = nn.Linear(joint_state_dim + joint_action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Critic 2 (Twin)
        self.l4 = nn.Linear(joint_state_dim + joint_action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, joint_state, joint_action):
        # Concatenate joint state and joint action
        xu = torch.cat([joint_state, joint_action], dim=1)
        
        # Critic 1 Forward
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        # Critic 2 Forward
        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)

        return x1, x2

    def Q1(self, joint_state, joint_action):
        """Returns only the first Q-value for actor updates."""
        xu = torch.cat([joint_state, joint_action], dim=1)
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1
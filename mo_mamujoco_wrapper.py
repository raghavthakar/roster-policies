import numpy as np
from gymnasium.spaces import Box

class MOMaMuJoCoWrapper:
    """
    A wrapper for MaMuJoCo PettingZoo environments to make them Multi-Objective.
    Does NOT inherit from gym.Wrapper to avoid type checks against gym.Env.
    """
    def __init__(self, env):
        self.env = env
        self.agents = self.env.agents
        self.possible_agents = self.env.possible_agents
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        
        # Define the reward space: 2 objectives per agent
        self.reward_dim = 2
        # Add reward_space attribute for compatibility
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(self.reward_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        """
        Execute actions, get info from base env, and reconstruct vector rewards.
        """
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        vector_rewards = {}
        
        for agent_id in rewards.keys():
            # Handle cases where agent might be done and info is empty or missing specific keys
            if agent_id not in infos:
                # If an agent is done, it might not have info. 
                # Return zero vector or handle gracefully.
                vector_rewards[agent_id] = np.zeros(self.reward_dim, dtype=np.float32)
                continue

            agent_info = infos[agent_id]
            
            # --- Extract Components ---
            # 1. Forward Progress (Velocity + Survive + Contact)
            # Default to 0.0 if key is missing (safety check)
            fwd_velocity = agent_info.get('reward_forward', 0.0)
            survive = agent_info.get('reward_survive', 0.0)
            contact_cost = agent_info.get('reward_contact', 0.0)
            
            obj_0 = fwd_velocity + survive + contact_cost

            # 2. Energy Efficiency (Control Cost)
            ctrl_cost = agent_info.get('reward_ctrl', 0.0)
            
            obj_1 = ctrl_cost

            # --- Construct Vector ---
            vector_rewards[agent_id] = np.array([obj_0, obj_1], dtype=np.float32)

        return obs, vector_rewards, terminations, truncations, infos

    def close(self):
        return self.env.close()

    def observation_space(self, agent):
        return self.env.observation_space(agent)

    def action_space(self, agent):
        return self.env.action_space(agent)

    def __getattr__(self, name):
        """Delegate any other attribute access to the underlying environment"""
        return getattr(self.env, name)
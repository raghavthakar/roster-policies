import numpy as np
import torch
from gymnasium_robotics.envs.multiagent_mujoco import mamujoco_v1
from moma_ac import MultiAgentTD3
from replay_buffer import MultiAgentReplayBuffer

# --- Hyperparameters ---
ENV_NAME = "Ant"
AGENT_CONF = "2x4"   # 2 agents, 4 legs each
SEED = 42
MAX_EPISODES = 5000
MAX_STEPS_PER_EP = 1000
BATCH_SIZE = 256
BUFFER_SIZE = 1_000_000
START_STEPS = 25_000   # Random steps before training begins
NOISE_SCALE = 0.1      # Exploration noise for actions
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"Initializing Environment: {ENV_NAME} ({AGENT_CONF}) on {DEVICE}")
    
    # 1. Initialize Environment
    # Note: parallel_env allows agents to step simultaneously
    env = mamujoco_v1.parallel_env(ENV_NAME, AGENT_CONF)
    obs, infos = env.reset(seed=SEED)

    # 2. Extract Agent Specifications
    # We need to dynamically get the obs/action dimensions for each agent
    agent_ids = env.agents
    obs_dims = {}
    act_dims = {}

    for agent in agent_ids:
        # Gymnasium spaces are typically Box(low, high, shape, dtype)
        obs_dims[agent] = env.observation_space(agent).shape[0]
        act_dims[agent] = env.action_space(agent).shape[0]

    print(f"Agents: {agent_ids}")
    print(f"Observation Dims: {obs_dims}")
    print(f"Action Dims: {act_dims}")

    # 3. Initialize Agent & Replay Buffer
    moma_agent = MultiAgentTD3(
        agent_ids=agent_ids,
        obs_dims=obs_dims,
        act_dims=act_dims,
        max_action=1.0,  # MuJoCo actions are usually [-1, 1]
        device=DEVICE
    )

    replay_buffer = MultiAgentReplayBuffer(
        max_size=BUFFER_SIZE,
        agent_ids=agent_ids,
        obs_dims=obs_dims,
        act_dims=act_dims,
        device=DEVICE
    )

    # 4. Main Training Loop
    total_steps = 0
    
    for episode in range(1, MAX_EPISODES + 1):
        obs, infos = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS_PER_EP):
            total_steps += 1
            
            # --- Select Action ---
            if total_steps < START_STEPS:
                # Warmup: Sample random actions from the environment space
                actions = {agent: env.action_space(agent).sample() for agent in agent_ids}
            else:
                # Training: Use policy with exploration noise
                actions = moma_agent.select_action(obs, noise_scale=NOISE_SCALE)

            # --- Step Environment ---
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # Prepare 'dones' (Termination or Truncation)
            # In MaMuJoCo, usually if one agent falls, the episode ends for all.
            dones = {
                agent: float(terminations[agent] or truncations[agent]) 
                for agent in agent_ids
            }

            # --- Store Transition ---
            replay_buffer.add(obs, actions, rewards, next_obs, dones)

            # --- Update State & Logging ---
            obs = next_obs
            # In cooperative MaMuJoCo, rewards are often identical or shared.
            # We assume a shared reward structure for the printout.
            episode_reward += rewards[agent_ids[0]]

            # --- Train Agent ---
            if total_steps >= START_STEPS:
                moma_agent.train(replay_buffer, batch_size=BATCH_SIZE)

            # Check for termination
            if any(dones.values()):
                break
        
        # Logging
        if episode % 10 == 0:
            print(f"Episode: {episode} | Steps: {total_steps} | Reward: {episode_reward:.2f}")

    # 5. Cleanup
    env.close()

if __name__ == "__main__":
    main()
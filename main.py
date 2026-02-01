import numpy as np
import torch
from gymnasium_robotics.envs.multiagent_mujoco import mamujoco_v1
from moma_ac import MultiAgentTD3
from replay_buffer import MultiAgentReplayBuffer
from mo_mamujoco_wrapper import MOMaMuJoCoWrapper

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
    env = MOMaMuJoCoWrapper(mamujoco_v1.parallel_env(ENV_NAME, AGENT_CONF))
    obs, infos = env.reset(seed=SEED)

    # 2. Extract Agent Specifications
    # Sort agent IDs to ensure consistent order for tensor concatenation across modules
    agent_ids = sorted(env.agents)
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
    # The preference dimension is 2 (Speed vs Energy) as defined in the wrapper
    preference_dim = 2 

    moma_agent = MultiAgentTD3(
        agent_ids=agent_ids,
        obs_dims=obs_dims,
        act_dims=act_dims,
        preference_dim=preference_dim,
        max_action=1.0,  # MuJoCo actions are usually [-1, 1]
        device=DEVICE
    )

    replay_buffer = MultiAgentReplayBuffer(
        max_size=BUFFER_SIZE,
        agent_ids=agent_ids,
        obs_dims=obs_dims,
        act_dims=act_dims,
        preference_dim=preference_dim,
        device=DEVICE
    )

    # 4. Main Training Loop
    total_steps = 0
    
    for episode in range(1, MAX_EPISODES + 1):
        # Sample preference w ~ U(0, 1) for this episode
        # In 2-obj case: w0 = rand, w1 = 1 - w0 (Paper Section 4.3)
        w0 = np.random.random()
        preference = np.array([w0, 1.0 - w0], dtype=np.float32)

        # For >2 objectives, use Dirichlet distribution instead:
        # preference = np.random.dirichlet(np.ones(preference_dim)).astype(np.float32)

        obs, infos = env.reset()
        
        # Initialize vector episode reward (2 objectives)
        episode_reward = np.zeros(2, dtype=np.float32)
        
        for step in range(MAX_STEPS_PER_EP):
            total_steps += 1
            
            # --- Select Action ---
            if total_steps < START_STEPS:
                # Warmup: Sample random actions from the environment space
                actions = {agent: env.action_space(agent).sample() for agent in agent_ids}
            else:
                # Training: Use policy with exploration noise conditioned on preference
                actions = moma_agent.select_action(obs, preference, noise_scale=NOISE_SCALE)

            # --- Step Environment ---
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # --- CRITICAL FIX: Handling Terminations vs Truncations ---
            # Terminations (robot fell/died) -> done=1 (Q-value should be 0)
            # Truncations (Time Limit) -> done=0 (Q-value should be bootstrapped)
            # If we set done=1 for Time Limits, we falsely tell the agent the future value is 0.
            
            dones_for_buffer = {
                agent: float(terminations[agent]) 
                for agent in agent_ids
            }

            # Check if episode should physically end (either died or ran out of time)
            # Note: In MaMuJoCo, if one agent terminates, usually all do.
            is_terminated = any(terminations.values())
            is_truncated = any(truncations.values())
            should_break = is_terminated or is_truncated

            # --- Store Transition ---
            # We pass 'dones_for_buffer' (contains only true failures) to the replay buffer
            replay_buffer.add(obs, actions, rewards, next_obs, dones_for_buffer, preference)

            # --- Update State & Logging ---
            obs = next_obs
            
            # Sum the vector rewards
            # In cooperative MaMuJoCo, rewards are identical for all agents.
            # We track this purely for logging purposes.
            episode_reward += rewards[agent_ids[0]]

            # --- Train Agent ---
            if total_steps >= START_STEPS:
                moma_agent.train(replay_buffer, batch_size=BATCH_SIZE)

            # Break loop if episode ended
            if should_break:
                break
        
        # Logging
        if episode % 10 == 0:
            # Format vector reward for printing
            rew_str = f"[{episode_reward[0]:.2f}, {episode_reward[1]:.2f}]"
            print(f"Episode: {episode} | Steps: {total_steps} | Preference: [{preference[0]:.2f}, {preference[1]:.2f}] | Reward Vector: {rew_str}")

    # 5. Cleanup
    env.close()

if __name__ == "__main__":
    main()
import numpy as np
import torch
import os
import random
import argparse
from gymnasium_robotics.envs.multiagent_mujoco import mamujoco_v1
from moma_ac import MultiAgentTD3
from replay_buffer import MultiAgentReplayBuffer
from mo_mamujoco_wrapper import MOMaMuJoCoWrapper

# --- Hyperparameters ---
ENV_NAME = "Ant"
AGENT_CONF = "2x4"
SEED = 42
MAX_EPISODES = 5000
MAX_STEPS_PER_EP = 1000
BATCH_SIZE = 256
BUFFER_SIZE = 1_000_000
START_STEPS = 25_000
NOISE_SCALE = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpointing settings
CHECKPOINT_DIR = "./checkpoints"
SAVE_INTERVAL_EPISODES = 50
LOAD_CHECKPOINT_PATH = "./checkpoints/latest_model.pth"
# LOAD_CHECKPOINT_PATH = None

def seed_everything(seed):
    """
    Sets seeds for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Force deterministic algorithms (slightly slower but necessary for identical runs)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to: {seed}")

def main():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # 1. Set Global Seed
    seed_everything(SEED)

    print(f"Initializing Environment: {ENV_NAME} ({AGENT_CONF}) on {DEVICE}")
    
    # Initialize Env
    env = MOMaMuJoCoWrapper(mamujoco_v1.parallel_env(ENV_NAME, AGENT_CONF))
    
    # Important: Seed the environment explicitly
    # Note: Gymnasium envs often have their own internal RNG that needs resetting.
    # We do the first reset with seed.
    obs, infos = env.reset(seed=SEED)

    agent_ids = sorted(env.agents)
    obs_dims = {}
    act_dims = {}

    for agent in agent_ids:
        obs_dims[agent] = env.observation_space(agent).shape[0]
        act_dims[agent] = env.action_space(agent).shape[0]

    preference_dim = 2 

    moma_agent = MultiAgentTD3(
        agent_ids=agent_ids,
        obs_dims=obs_dims,
        act_dims=act_dims,
        preference_dim=preference_dim,
        max_action=1.0,
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

    # --- Resume Logic ---
    start_episode = 1
    total_steps = 0

    if LOAD_CHECKPOINT_PATH and os.path.exists(LOAD_CHECKPOINT_PATH):
        print(f"Resuming training from: {LOAD_CHECKPOINT_PATH}")
        moma_agent.load(LOAD_CHECKPOINT_PATH)
        
        # Load metadata and RNG states
        meta_path = LOAD_CHECKPOINT_PATH.replace("_model.pth", "_meta.npy")
        if os.path.exists(meta_path):
            meta_data = np.load(meta_path, allow_pickle=True).item()
            start_episode = meta_data.get('episode', 1) + 1
            total_steps = meta_data.get('total_steps', 0)
            
            # --- CRITICAL: Restore RNG States ---
            if 'rng_states' in meta_data:
                print("Restoring RNG states for deterministic resume...")
                rng_states = meta_data['rng_states']
                random.setstate(rng_states['python'])
                np.random.set_state(rng_states['numpy'])
                torch.set_rng_state(rng_states['torch'])
                if torch.cuda.is_available() and 'torch_cuda' in rng_states:
                    torch.cuda.set_rng_state_all(rng_states['torch_cuda'])
                
                # Try to restore environment RNG if possible (Best Effort)
                # Note: Gymnasium RNG is tricky to restore perfectly without direct access
                # to the internal generator, but seeding handled the initial divergence.
                # For strict resume, relying on the global np.random state is key 
                # if the env relies on it (which many do indirectly via inputs).

            print(f"Resuming from Episode {start_episode}, Step {total_steps}")
        
        buffer_path = os.path.join(os.path.dirname(LOAD_CHECKPOINT_PATH), "latest_buffer.npz")
        if os.path.exists(buffer_path):
            replay_buffer.load_state(buffer_path)

    # --- Training Loop ---
    for episode in range(start_episode, MAX_EPISODES + 1):
        
        # Deterministic preference sampling (relies on np.random state)
        w0 = np.random.random()
        preference = np.array([w0, 1.0 - w0], dtype=np.float32)

        # Standard reset (uses env's internal RNG, or global if not isolated)
        obs, infos = env.reset()
        episode_reward = np.zeros(2, dtype=np.float32)
        
        for step in range(MAX_STEPS_PER_EP):
            total_steps += 1
            
            if total_steps < START_STEPS and not LOAD_CHECKPOINT_PATH:
                actions = {agent: env.action_space(agent).sample() for agent in agent_ids}
            else:
                actions = moma_agent.select_action(obs, preference, noise_scale=NOISE_SCALE)

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            dones_for_buffer = {
                agent: float(terminations[agent]) 
                for agent in agent_ids
            }

            replay_buffer.add(obs, actions, rewards, next_obs, dones_for_buffer, preference)

            obs = next_obs
            episode_reward += rewards[agent_ids[0]]

            if total_steps >= START_STEPS:
                moma_agent.train(replay_buffer, batch_size=BATCH_SIZE)

            if any(terminations.values()) or any(truncations.values()):
                break
        
        if episode % 10 == 0:
            rew_str = f"[{episode_reward[0]:.2f}, {episode_reward[1]:.2f}]"
            print(f"Episode: {episode} | Steps: {total_steps} | Preference: [{preference[0]:.2f}, {preference[1]:.2f}] | Reward: {rew_str}")

        # --- Checkpointing ---
        if episode % SAVE_INTERVAL_EPISODES == 0:
            # Prepare RNG states
            rng_states = {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            }

            # 1. Save Periodic Model (Lightweight)
            hist_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{episode}_model.pth")
            moma_agent.save(hist_path)

            # 2. Save Latest (Heavy)
            latest_model_path = os.path.join(CHECKPOINT_DIR, "latest_model.pth")
            latest_meta_path = os.path.join(CHECKPOINT_DIR, "latest_meta.npy")
            latest_buffer_path = os.path.join(CHECKPOINT_DIR, "latest_buffer.npz")
            
            moma_agent.save(latest_model_path)
            
            # Save Meta with RNG
            np.save(latest_meta_path, {
                'episode': episode, 
                'total_steps': total_steps,
                'rng_states': rng_states
            })
            
            replay_buffer.save_state(latest_buffer_path)

    env.close()

if __name__ == "__main__":
    main()
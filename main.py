import numpy as np
import torch
import os
import random
import argparse
from gymnasium_robotics.envs.multiagent_mujoco import mamujoco_v1
from moma_ac import MultiAgentTD3
from replay_buffer import MultiAgentReplayBuffer
from mo_mamujoco_wrapper import MOMaMuJoCoWrapper

# --- Default Constants ---
ENV_NAME = "Ant"
AGENT_CONF = "2x4"
MAX_STEPS = 500_000        # Total environment steps to train for
MAX_STEPS_PER_EP = 1000
BATCH_SIZE = 256
BUFFER_SIZE = 1_000_000
START_STEPS = 25_000
NOISE_SCALE = 0.1
SAVE_INTERVAL_STEPS = 50_000  # Save checkpoint every 50k steps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- MoE Settings ---
NUM_EXPERTS = 4         
MOE_LAMBDA = 0.01       

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
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to: {seed}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train MOMA-AC Agents")
    
    # 1. Use MOE (Flag)
    parser.add_argument('--use_moe', action='store_true', default=False,
                        help='Enable Mixture of Experts Architecture')
    
    # 2. Checkpoint Base Directory
    parser.add_argument('--checkpoint_base_dir', type=str, default='./checkpoints',
                        help='Base directory to store training runs')
    
    # 3. Seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()

def main():
    args = parse_args()
    
    # Construct Run Name & Directory
    algo_name = "MOMA_MoE" if args.use_moe else "MOMA_Baseline"
    run_name = f"{algo_name}_seed_{args.seed}"
    save_dir = os.path.join(args.checkpoint_base_dir, run_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"--- Starting Run: {run_name} ---")
    print(f"Saving to: {save_dir}")

    # 1. Set Global Seed
    seed_everything(args.seed)

    print(f"Initializing Environment: {ENV_NAME} ({AGENT_CONF}) on {DEVICE}")
    
    # Initialize Env
    env = MOMaMuJoCoWrapper(mamujoco_v1.parallel_env(ENV_NAME, AGENT_CONF))
    obs, infos = env.reset(seed=args.seed)

    agent_ids = sorted(env.agents)
    obs_dims = {}
    act_dims = {}

    for agent in agent_ids:
        obs_dims[agent] = env.observation_space(agent).shape[0]
        act_dims[agent] = env.action_space(agent).shape[0]

    preference_dim = 2 

    # Initialize Agent
    moma_agent = MultiAgentTD3(
        agent_ids=agent_ids,
        obs_dims=obs_dims,
        act_dims=act_dims,
        preference_dim=preference_dim,
        max_action=1.0,
        device=DEVICE,
        use_moe=args.use_moe,
        num_experts=NUM_EXPERTS,
        moe_lambda=MOE_LAMBDA
    )

    replay_buffer = MultiAgentReplayBuffer(
        max_size=BUFFER_SIZE,
        agent_ids=agent_ids,
        obs_dims=obs_dims,
        act_dims=act_dims,
        preference_dim=preference_dim,
        device=DEVICE
    )

    # --- Resume Logic (Optional) ---
    # To implement resume properly with step-based names, you'd check save_dir for "latest_model.pth"
    latest_path = os.path.join(save_dir, "latest_model.pth")
    total_steps = 0
    episode_count = 0

    if os.path.exists(latest_path):
        print(f"Resuming training from: {latest_path}")
        moma_agent.load(latest_path)
        
        meta_path = os.path.join(save_dir, "latest_meta.npy")
        if os.path.exists(meta_path):
            meta_data = np.load(meta_path, allow_pickle=True).item()
            total_steps = meta_data.get('total_steps', 0)
            episode_count = meta_data.get('episode', 0)
            
            # Restore RNG
            if 'rng_states' in meta_data:
                rng_states = meta_data['rng_states']
                random.setstate(rng_states['python'])
                np.random.set_state(rng_states['numpy'])
                torch.set_rng_state(rng_states['torch'])
                if torch.cuda.is_available() and 'torch_cuda' in rng_states:
                    torch.cuda.set_rng_state_all(rng_states['torch_cuda'])
        
        buffer_path = os.path.join(save_dir, "latest_buffer.npz")
        if os.path.exists(buffer_path) and hasattr(replay_buffer, 'load_state'):
            replay_buffer.load_state(buffer_path)

    # --- Step-Based Training Loop ---
    print(f"Training Start: {total_steps} / {MAX_STEPS} steps")
    
    # Calculate next checkpoint threshold
    next_save_threshold = (total_steps // SAVE_INTERVAL_STEPS + 1) * SAVE_INTERVAL_STEPS

    while total_steps < MAX_STEPS:
        episode_count += 1
        
        # New Episode Setup
        w0 = np.random.random()
        preference = np.array([w0, 1.0 - w0], dtype=np.float32)
        obs, infos = env.reset()
        episode_reward = np.zeros(2, dtype=np.float32)
        
        for step in range(MAX_STEPS_PER_EP):
            total_steps += 1
            
            # Action Selection
            if total_steps < START_STEPS and not os.path.exists(latest_path):
                actions = {agent: env.action_space(agent).sample() for agent in agent_ids}
            else:
                actions = moma_agent.select_action(obs, preference, noise_scale=NOISE_SCALE)

            # Step
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            dones_for_buffer = {
                agent: float(terminations[agent]) 
                for agent in agent_ids
            }

            replay_buffer.add(obs, actions, rewards, next_obs, dones_for_buffer, preference)

            obs = next_obs
            episode_reward += rewards[agent_ids[0]]

            # Train
            if total_steps >= START_STEPS:
                moma_agent.train(replay_buffer, batch_size=BATCH_SIZE)

            # --- Step-Based Checkpointing ---
            if total_steps >= next_save_threshold:
                print(f"--> Saving Checkpoint at {total_steps} steps...")
                
                rng_states = {
                    'python': random.getstate(),
                    'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(),
                    'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                }

                # 1. Periodic Checkpoint (e.g., checkpoint_50000_model.pth)
                hist_path = os.path.join(save_dir, f"checkpoint_{total_steps}_model.pth")
                moma_agent.save(hist_path)

                # 2. Latest Checkpoint (Overwrites)
                moma_agent.save(os.path.join(save_dir, "latest_model.pth"))
                
                np.save(os.path.join(save_dir, "latest_meta.npy"), {
                    'episode': episode_count, 
                    'total_steps': total_steps,
                    'rng_states': rng_states
                })
                
                if hasattr(replay_buffer, 'save_state'):
                    replay_buffer.save_state(os.path.join(save_dir, "latest_buffer.npz"))

                # Update next threshold
                next_save_threshold += SAVE_INTERVAL_STEPS

            # Termination Logic
            if any(terminations.values()) or any(truncations.values()):
                break
            
            # Stop if max steps reached mid-episode
            if total_steps >= MAX_STEPS:
                break
        
        # Logging (Per Episode)
        if episode_count % 10 == 0:
            rew_str = f"[{episode_reward[0]:.2f}, {episode_reward[1]:.2f}]"
            print(f"Ep: {episode_count} | Steps: {total_steps}/{MAX_STEPS} | Pref: [{preference[0]:.2f}, {preference[1]:.2f}] | Rew: {rew_str}")

    print("Training Complete.")
    env.close()

if __name__ == "__main__":
    main()
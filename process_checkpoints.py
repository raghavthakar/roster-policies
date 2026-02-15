import torch
import numpy as np
import os
import argparse
import pickle
import re
from tqdm import tqdm

from gymnasium_robotics.envs.multiagent_mujoco import mamujoco_v1
from moma_ac import MultiAgentTD3
from mo_mamujoco_wrapper import MOMaMuJoCoWrapper

# --- Configuration ---
ENV_NAME = "Ant"
AGENT_CONF = "2x4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EVAL_POINTS = 100
FIXED_EVAL_SEED = 1337 

def get_agent_architecture(checkpoint_path):
    """
    Peeks into the checkpoint to determine if it uses MoE or Standard architecture.
    Returns: (use_moe, num_experts, checkpoint_dict)
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        actor_state = checkpoint['actor']
        
        # Check for MoE specific keys
        use_moe = any("routers" in k for k in actor_state.keys())
        
        # Determine num_experts if MoE
        num_experts = 3 # Default fallback
        if use_moe:
            # Try to infer expert count from the shape of a router bias
            for k, v in actor_state.items():
                if "routers" in k and "bias" in k and k.endswith(".2.bias"):
                    num_experts = v.shape[0]
                    break
        
        return use_moe, num_experts, checkpoint
    except Exception as e:
        print(f"Error inspecting {checkpoint_path}: {e}")
        return False, 3, None

def evaluate_model(agent, env, num_points=100):
    """
    Rolls out the policy for 'num_points' distinct preferences.
    Returns: 
        preferences: (num_points, 2) array of input prefs
        rewards: (num_points, 2) array of resulting cumulative rewards
    """
    preference_list = []
    reward_list = []
    
    # Linear interpolation from [0.01, 0.99] to [0.99, 0.01]
    w0_values = np.linspace(0.01, 0.99, num_points)
    
    for i, w0 in enumerate(w0_values):
        w1 = 1.0 - w0
        preference = np.array([w0, w1], dtype=np.float32)
        
        # CRITICAL: Seed each episode identically across models/runs for fairness.
        episode_seed = FIXED_EVAL_SEED + i
        
        obs, _ = env.reset(seed=episode_seed)
        episode_reward = np.zeros(2, dtype=np.float32)
        
        # Rollout
        # We assume max steps (1000) is sufficient to finish or truncate
        max_steps = 1000
        
        for _ in range(max_steps):
            # Select action (Deterministic for eval)
            actions = agent.select_action(obs, preference, noise_scale=0.0)
            
            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            
            # Cooperative reward: take agent 0's reward (shared)
            agent_0 = env.agents[0]
            episode_reward += rewards[agent_0]
            
            obs = next_obs
            
            if any(terminations.values()) or any(truncations.values()):
                break
        
        preference_list.append(preference)
        reward_list.append(episode_reward)
        
    return np.array(preference_list), np.array(reward_list)

def extract_step_from_filename(filename):
    """Extracts the timestep number from 'checkpoint_50000_model.pth'"""
    # Look for integers in the filename
    match = re.search(r'checkpoint_(\d+)_model', filename)
    if match:
        return int(match.group(1))
    # Fallback if filename is just 'latest_model.pth' -> we rely on metadata inside later
    if "latest" in filename:
        return -1 
    return 0

def parse_run_directory(dirname):
    """
    Parses 'MOMA_Baseline_seed_1' into ('MOMA_Baseline', 1)
    """
    # Regex to find the last occurrence of '_seed_' and split there
    match = re.search(r'(.+)_seed_(\d+)$', dirname)
    if match:
        algo_name = match.group(1)
        seed = int(match.group(2))
        return algo_name, seed
    return dirname, 0 # Fallback

def main():
    parser = argparse.ArgumentParser(description="Process MOMA-AC Checkpoints to Raw Data")
    parser.add_argument('base_dir', type=str, help='Base directory containing run folders (e.g. ./checkpoints)')
    parser.add_argument('--output', type=str, default='evaluation_raw_data.pkl', help='Output pickle filename')
    args = parser.parse_args()

    # 1. Setup Environment (Created once)
    print(f"Initializing Environment: {ENV_NAME} {AGENT_CONF}")
    base_env = mamujoco_v1.parallel_env(ENV_NAME, AGENT_CONF)
    env = MOMaMuJoCoWrapper(base_env)
    
    # Get Dims
    obs_dims = {a: env.observation_space(a).shape[0] for a in env.agents}
    act_dims = {a: env.action_space(a).shape[0] for a in env.agents}
    agent_ids = sorted(env.agents)
    preference_dim = 2

    # Data Structure: results[algo_name][seed][timestep] = {'prefs': ..., 'rewards': ...}
    results = {}

    # 2. Scan Directories
    if not os.path.exists(args.base_dir):
        print(f"Error: Directory {args.base_dir} does not exist.")
        return

    subdirs = [d for d in os.listdir(args.base_dir) if os.path.isdir(os.path.join(args.base_dir, d))]
    
    if not subdirs:
        print("No subdirectories found.")
        return

    print(f"Found {len(subdirs)} run directories.")

    for subdir in subdirs:
        full_subdir_path = os.path.join(args.base_dir, subdir)
        algo_name, seed = parse_run_directory(subdir)
        
        print(f"\nProcessing Run: {algo_name} | Seed: {seed}")
        
        # Initialize dictionary structure
        if algo_name not in results:
            results[algo_name] = {}
        if seed not in results[algo_name]:
            results[algo_name][seed] = {}

        # Find .pth files
        checkpoints = [f for f in os.listdir(full_subdir_path) if f.endswith(".pth") and "model" in f]
        
        # Sort by timestep for clean processing order
        # We assume format 'checkpoint_X_model.pth'. 
        # Note: 'latest_model.pth' might duplicate the last checkpoint step. 
        # We will check timestamps inside to confirm.
        checkpoints.sort(key=extract_step_from_filename)
        
        for ckpt in tqdm(checkpoints, desc=f"  Eval Checkpoints"):
            full_ckpt_path = os.path.join(full_subdir_path, ckpt)
            
            # A. Peek at architecture & Load
            use_moe, num_experts, checkpoint_data = get_agent_architecture(full_ckpt_path)
            
            # B. Get Timestep (Prefer internal counter, fallback to filename)
            timestep = checkpoint_data.get('total_it', 0)
            if timestep == 0:
                timestep = extract_step_from_filename(ckpt)
                
            # If we already processed this timestep (e.g. latest_model == checkpoint_500k), skip
            if timestep in results[algo_name][seed]:
                continue

            # C. Initialize Agent
            agent = MultiAgentTD3(
                agent_ids=agent_ids,
                obs_dims=obs_dims,
                act_dims=act_dims,
                preference_dim=preference_dim,
                max_action=1.0,
                device=DEVICE,
                use_moe=use_moe,
                num_experts=num_experts
            )
            agent.actor.load_state_dict(checkpoint_data['actor'])
            
            # D. Run Evaluation
            prefs, rewards = evaluate_model(agent, env, num_points=NUM_EVAL_POINTS)
            
            # E. Store Raw Data
            results[algo_name][seed][timestep] = {
                'preferences': prefs,
                'rewards': rewards
            }

    # 3. Save to Disk
    print(f"\nSaving raw results to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    
    print("Done. Data saved successfully.")
    env.close()

if __name__ == "__main__":
    main()
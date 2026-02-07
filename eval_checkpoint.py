import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from gymnasium_robotics.envs.multiagent_mujoco import mamujoco_v1
from moma_ac import MultiAgentTD3
from mo_mamujoco_wrapper import MOMaMuJoCoWrapper

# --- Configuration ---
ENV_NAME = "Ant"
AGENT_CONF = "2x4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REF_POINT = np.array([-1000.0, -1000.0]) # Reference point for HV calculation (from paper)
NUM_EVAL_POINTS = 100
FIXED_EVAL_SEED = 1337 

def calculate_hypervolume_2d(points, ref_point):
    """
    Calculates 2D Hypervolume dominated by points w.r.t ref_point.
    Points should be maximized (higher is better).
    """
    # 1. Filter points that do not dominate the reference point
    valid_points = [p for p in points if np.all(p > ref_point)]
    if not valid_points:
        return 0.0
    
    valid_points = np.array(valid_points)

    # 2. Sort by 1st objective (ascending)
    # For 2D, if we sort by X, the Y values of the Pareto front must be descending.
    sorted_indices = np.argsort(valid_points[:, 0])
    sorted_points = valid_points[sorted_indices]

    # 3. Filter to keep only the Pareto frontier
    # (Remove dominated points to simplify calculation)
    pareto_front = []
    current_max_y = -np.inf
    
    # Iterate backwards (highest X first)
    for point in sorted_points[::-1]:
        if point[1] > current_max_y:
            pareto_front.append(point)
            current_max_y = point[1]
    
    # Now pareto_front is sorted by X descending
    pareto_front = np.array(pareto_front)

    # 4. Calculate Area
    # Sum of rectangles: (X_i - X_{ref}) * (Y_i - Y_{next})
    # Since it's sorted by X descending:
    # Top-right point forms rectangle with ref X and previous Y.
    
    area = 0.0
    previous_y = ref_point[1]
    
    # We iterate from right (highest X) to left (lowest X)
    for x, y in pareto_front:
        width = x - ref_point[0]
        height = y - previous_y
        if width > 0 and height > 0:
            area += width * height
        previous_y = y # The "floor" rises to the current point's Y
        
    return area

def get_agent_architecture(checkpoint_path):
    """
    Peeks into the checkpoint to determine if it uses MoE or Standard architecture.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        actor_state = checkpoint['actor']
        
        # Check for MoE specific keys
        use_moe = any("routers" in k for k in actor_state.keys())
        
        # Determine num_experts if MoE
        num_experts = 3 # Default
        if use_moe:
            # Try to infer expert count from the shape of a router weight
            # routers.agent_0.2.bias shape is [num_experts]
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
    Rolls out the policy for 100 distinct preferences.
    Returns: A list of result vectors.
    """
    results = []
    
    # Linear interpolation from [0.01, 0.99] to [0.99, 0.01]
    # Avoiding 0.0/1.0 to prevent numerical instabilities in some log calculations
    w0_values = np.linspace(0.01, 0.99, num_points)
    
    for i, w0 in enumerate(w0_values):
        w1 = 1.0 - w0
        preference = np.array([w0, w1], dtype=np.float32)
        
        # CRITICAL: Seed each episode identically across models/runs for fairness.
        # We use a unique seed for each preference point, but consistent across checkpoints.
        episode_seed = FIXED_EVAL_SEED + i
        
        obs, _ = env.reset(seed=episode_seed)
        episode_reward = np.zeros(2, dtype=np.float32)
        
        # Rollout
        # Note: We assume max steps is sufficient to finish or truncate
        # Using environment's internal max step if defined, else 1000
        max_steps = 1000
        
        for _ in range(max_steps):
            # Select action (Deterministic for eval)
            # noise_scale=0.0 ensures we evaluate the policy capability, not luck
            actions = agent.select_action(obs, preference, noise_scale=0.0)
            
            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            
            # Cooperative reward: take agent 0
            agent_0 = env.agents[0]
            episode_reward += rewards[agent_0]
            
            obs = next_obs
            
            if any(terminations.values()) or any(truncations.values()):
                break
                
        results.append(episode_reward)
        
    return np.array(results)

def main():
    parser = argparse.ArgumentParser(description="Evaluate MOMA-AC Checkpoints")
    parser.add_argument('dirs', nargs='+', help='List of directories containing .pth checkpoints')
    parser.add_argument('--output', type=str, default='hypervolume_plot.png', help='Output filename for plot')
    args = parser.parse_args()

    # Setup Environment
    print(f"Initializing Environment: {ENV_NAME} {AGENT_CONF}")
    base_env = mamujoco_v1.parallel_env(ENV_NAME, AGENT_CONF)
    env = MOMaMuJoCoWrapper(base_env)
    
    # Get Dims
    obs_dims = {a: env.observation_space(a).shape[0] for a in env.agents}
    act_dims = {a: env.action_space(a).shape[0] for a in env.agents}
    agent_ids = sorted(env.agents)
    preference_dim = 2

    # Plotting Data Storage
    algo_results = {}

    for directory in args.dirs:
        if not os.path.isdir(directory):
            print(f"Skipping {directory}, not a folder.")
            continue
            
        print(f"Processing Directory: {directory}")
        algo_name = os.path.basename(os.path.normpath(directory))
        
        # Find checkpoints
        checkpoints = [f for f in os.listdir(directory) if f.endswith(".pth") and "model" in f]
        
        # Try to sort by timestep integer in filename (assuming format "checkpoint_100_model.pth" or similar)
        # Helper to extract number
        def extract_step(name):
            parts = name.replace(".pth", "").split("_")
            for p in parts:
                if p.isdigit():
                    return int(p)
            return 0
            
        checkpoints.sort(key=extract_step)
        
        x_timesteps = []
        y_hypervolumes = []
        
        for ckpt in tqdm(checkpoints, desc=f"Eval {algo_name}"):
            full_path = os.path.join(directory, ckpt)
            
            # 1. Peek at architecture
            use_moe, num_experts, checkpoint_data = get_agent_architecture(full_path)
            
            # 2. Initialize Agent
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
            
            # 3. Load Weights
            agent.actor.load_state_dict(checkpoint_data['actor'])
            # We don't strictly need critics for evaluation, but the class inits them.
            
            # 4. Get Timestep
            timestep = checkpoint_data.get('total_it', 0)
            if timestep == 0:
                timestep = extract_step(ckpt)
                
            # 5. Rollout
            frontier_points = evaluate_model(agent, env, num_points=NUM_EVAL_POINTS)
            
            # 6. Calculate HV
            hv = calculate_hypervolume_2d(frontier_points, REF_POINT)
            
            x_timesteps.append(timestep)
            y_hypervolumes.append(hv)
            
        algo_results[algo_name] = (x_timesteps, y_hypervolumes)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    for algo, (steps, hvs) in algo_results.items():
        # Sort by steps just in case
        zipped = sorted(zip(steps, hvs))
        steps = [z[0] for z in zipped]
        hvs = [z[1] for z in zipped]
        
        plt.plot(steps, hvs, marker='o', label=algo)
        
    plt.title(f"Hypervolume vs Timestep ({ENV_NAME} {AGENT_CONF})")
    plt.xlabel("Training Steps")
    plt.ylabel("Hypervolume")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    print(f"Saving plot to {args.output}")
    plt.savefig(args.output)
    env.close()

if __name__ == "__main__":
    main()
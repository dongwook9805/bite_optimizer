import gymnasium as gym
import sys
import os
import numpy as np

# Add project root to path so we can import envs
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import envs # Registers the environment

def test_env():
    print("Initializing Environment...")
    # Use absolute paths for assets
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    assets_dir = os.path.join(project_root, 'assets')
    
    maxilla_path = os.path.join(assets_dir, 'dummy_maxilla.stl')
    mandible_path = os.path.join(assets_dir, 'dummy_mandible.stl')
    
    # Check if they exist, if not, warn or create?
    if not os.path.exists(maxilla_path):
        print("Assets not found, please run test_simulator.py first to generate them.")
        return

    env = gym.make('BiteOptimizer-v0', maxilla_path=maxilla_path, mandible_path=mandible_path)
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial Obs: {obs}")
    
    print("\nRunning Random Episode...")
    done = False
    total_reward = 0
    step = 0
    
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step += 1
        
        if step % 10 == 0:
            print(f"Step {step}: Reward={reward:.4f}, ContactScore={info.get('contact_score', 0):.4f}")
            
        if terminated or truncated:
            done = True
            
    print(f"\nEpisode finished after {step} steps. Total Reward: {total_reward:.4f}")

if __name__ == "__main__":
    test_env()

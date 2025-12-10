import os
import sys
import numpy as np

# Ensure src and envs are importable
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../envs'))

from bite_env import BiteEnv

def verify_icp_integration():
    assets_dir = os.path.join(os.path.dirname(__file__), '../assets')
    maxilla_path = os.path.join(assets_dir, 'dummy_maxilla.stl')
    mandible_path = os.path.join(assets_dir, 'dummy_mandible.stl')
    
    # Test WITHOUT ICP
    print("--- Testing WITHOUT ICP ---")
    env_no_icp = BiteEnv(maxilla_path, mandible_path, use_icp=False)
    obs, info = env_no_icp.reset()
    start_trans_no_icp = env_no_icp.simulator.current_translation.copy()
    print(f"Start translation (No ICP): {start_trans_no_icp}")
    
    # Test WITH ICP
    print("\n--- Testing WITH ICP ---")
    env_icp = BiteEnv(maxilla_path, mandible_path, use_icp=True)
    obs, info = env_icp.reset()
    start_trans_icp = env_icp.simulator.current_translation.copy()
    
    from utils import orthodontic_occlusion_reward
    start_metrics = env_icp.simulator.get_orthodontic_metrics()
    start_reward = orthodontic_occlusion_reward(**start_metrics)
    
    print(f"Start translation (With ICP): {start_trans_icp}")
    print(f"Start Reward (With ICP): {start_reward}")
    
    # Test RL Inference if model exists
    model_path = os.path.join(os.path.dirname(__file__), '../models/ppo_bite.zip')
    if os.path.exists(model_path):
        print("\n--- Testing RL Fine-tuning ---")
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
            
            # Observe initial state (already reset with ICP)
            curr_obs = obs
            total_reward = 0
            
            print("Running RL steps...")
            for i in range(50):
                action, _ = model.predict(curr_obs, deterministic=True)
                curr_obs, reward, terminated, truncated, _ = env_icp.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            
            end_trans = env_icp.simulator.current_translation.copy()
            end_metrics = env_icp.simulator.get_orthodontic_metrics()
            end_reward = orthodontic_occlusion_reward(**end_metrics)
            
            print(f"End translation (After RL): {end_trans}")
            print(f"End Reward (After RL): {end_reward}")
            print(f"Improvement: {end_reward - start_reward}")
            
        except ImportError:
            print("Stable Baselines 3 not installed provided in env.")
        except Exception as e:
            print(f"Error running RL: {e}")
    else:
        print("\nNo RL model found at models/ppo_bite.zip. Skipping RL test.")

if __name__ == "__main__":
    verify_icp_integration()

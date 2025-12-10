import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
import argparse
from stable_baselines3.common.env_util import make_vec_env

# Ensure src and envs are importable
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../envs'))

from bite_env import BiteEnv

def train(steps=10000, save_path="models/ppo_bite"):
    print(f"Starting training for {steps} steps...")
    
    # Check assets
    assets_dir = os.path.join(os.path.dirname(__file__), '../assets')
    maxilla_path = os.path.join(assets_dir, 'dummy_maxilla.stl')
    mandible_path = os.path.join(assets_dir, 'dummy_mandible.stl')
    
    if not os.path.exists(maxilla_path) or not os.path.exists(mandible_path):
        print("Assets not found! Run generate_dental_assets.py first.")
        return

    # Create Env
    # Use make_vec_env for potential parallel training, or just standard wrapping
    # We pass use_icp=True to train on FINE-TUNING from roughly aligned state
    env_kwargs = {
        'maxilla_path': maxilla_path, 
        'mandible_path': mandible_path, 
        'use_icp': True
    }
    
    # We can use a single environment for simplicity
    env = BiteEnv(**env_kwargs)
    
    # Initialize PPO
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
    
    # Train
    model.learn(total_timesteps=steps)
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000, help="Total training timesteps")
    parser.add_argument("--out", type=str, default="models/ppo_bite", help="Output model path")
    args = parser.parse_args()
    
    train(steps=args.steps, save_path=args.out)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys

# Ensure src is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from simulator import BiteSimulator
from utils import reward_function

class BiteEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, maxilla_path, mandible_path, render_mode=None):
        super(BiteEnv, self).__init__()
        
        self.render_mode = render_mode
        self.simulator = BiteSimulator(maxilla_path, mandible_path)
        
        # Action Space: [dRx, dRy, dRz, dTx, dTy, dTz]
        # Normalized to [-1, 1], scaled inside step function
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # Action scaling factors (per step)
        # User suggestion: Rotation ±0.1 deg, Translation ±0.05 mm
        self.max_rot = 0.5  # degrees (increased slightly for faster exploration?) let's stick to small
        self.max_trans = 0.1 # mm
        
        # Observation Space:
        # [Rx, Ry, Rz, Tx, Ty, Tz, contact_score, balance, penetration, ... ]
        # Let's say size 9 for now: 6 pose + 3 metrics
        # Bounds? Pose can be anything roughly. Metrics can be anything.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        self.steps = 0
        self.max_steps = 100
        
        # Keep track of cumulative pose relative to start
        # Simulator keeps state, but we might want to expose it explicitly
        
        self.last_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        
        # Reset simulator
        self.simulator.reset()
        
        # Optional: Randomize initial state slightly for RL robustness
        # For now, start from "Near Optimal" or "Standard" position?
        # User said: "Initial mandible pose = random small perturbation"
        
        initial_perturb_r = self.np_random.uniform(-2, 2, size=3) # +/- 2 degrees
        initial_perturb_t = self.np_random.uniform(-1, 1, size=3) # +/- 1 mm
        
        self.simulator.apply_transform(initial_perturb_r, initial_perturb_t)
        
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.steps += 1
        
        # Scale action
        delta_r = action[:3] * self.max_rot
        delta_t = action[3:] * self.max_trans
        
        # Apply physics
        self.simulator.apply_transform(delta_r, delta_t)
        
        # Compute metrics
        metrics = self.simulator.get_metrics()
        
        # Calculate Reward
        # We can use "Delta Reward" (improvement) or "Absolute Reward"
        # User mentioned: "reward_t = f(R,T)" is fine.
        current_reward = reward_function(metrics['distances'], metrics['points'])
        
        # Calculate delta if desired, but PPO works fine with absolute reward usually 
        # as long as we want to maximize cumulative return. 
        # But if we want to find "The Best Pose", dense reward based on score is good.
        
        reward = current_reward
        
        # Terminate?
        terminated = False
        truncated = False
        
        if self.steps >= self.max_steps:
            truncated = True
            
        # Optional: Early stopping if moved too far away?
        
        obs = self._get_obs(metrics, current_reward)
        info = {
            "contact_score": current_reward # placeholder breakdown
        }
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self, metrics=None, current_reward=0.0):
        # Construct state vector
        # Pose (6) + Metrics
        # Simulator tracks current_translation. Rotation is implicit in mesh.
        # We should probably track accumulated rotation in simulator or here.
        # For now, let's just use translation (3) + dummy rotation (3) + metrics.
        # Ideally simulator should return current pose params.
        
        # Since simulator.current_translation is available
        trans = self.simulator.current_translation
        rot = np.zeros(3) # TODO: Track rotation properly in simulator
        
        if metrics is None:
            metrics = self.simulator.get_metrics()
            current_reward = reward_function(metrics['distances'], metrics['points'])
            
        # Metrics summary
        # contact_count
        distances = metrics['distances']
        contacts = np.sum((distances <= 0.1) & (distances > 0)) # approx
        penetration = np.sum(distances[distances < 0]) # magnitude
        
        return np.concatenate([
            rot, 
            trans, 
            [current_reward, contacts, penetration]
        ]).astype(np.float32)

    def render(self):
        # Visualization could go here.
        # For now, simulator doesn't handle visuals window.
        pass

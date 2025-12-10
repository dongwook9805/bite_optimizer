import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys

# Ensure src is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from simulator import BiteSimulator
from utils import orthodontic_occlusion_reward

class BiteEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, maxilla_path, mandible_path, render_mode=None, use_icp=True):
        super(BiteEnv, self).__init__()
        
        self.render_mode = render_mode
        self.use_icp = use_icp
        self.simulator = BiteSimulator(maxilla_path, mandible_path)
        
        # Action Space: [dRx, dRy, dRz, dTx, dTy, dTz]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        self.max_rot = 0.5 
        self.max_trans = 0.1 
        
        # Observation Space:
        # [Pose(6) + OrthoMetrics(12)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        
        self.steps = 0
        self.max_steps = 2000
        self.last_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        
        self.simulator.reset()
        if self.use_icp:
            self.simulator.rough_align_icp()
        
        initial_perturb_r = self.np_random.uniform(-2, 2, size=3)
        initial_perturb_t = self.np_random.uniform(-1, 1, size=3)
        self.simulator.apply_transform(initial_perturb_r, initial_perturb_t)
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.steps += 1
        
        delta_r = action[:3] * self.max_rot
        delta_t = action[3:] * self.max_trans
        self.simulator.apply_transform(delta_r, delta_t)
        
        # Calculate Reward
        ortho_metrics = self.simulator.get_orthodontic_metrics()
        current_reward = orthodontic_occlusion_reward(**ortho_metrics)
        
        reward = current_reward
        
        terminated = False
        truncated = self.steps >= self.max_steps
        
        metrics_dict = self.simulator.get_metrics() # Still needed for info/obs?
        # Actually orthodontic_metrics is better.
        
        obs = self._get_obs(ortho_metrics, current_reward)
        info = ortho_metrics
        info["reward"] = current_reward
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self, ortho_metrics=None, current_reward=0.0):
        trans = self.simulator.current_translation
        rot = np.zeros(3) 
        
        if ortho_metrics is None:
            ortho_metrics = self.simulator.get_orthodontic_metrics()
            current_reward = orthodontic_occlusion_reward(**ortho_metrics)
            
        # Extract values in order
        m_vals = [
            ortho_metrics["overjet_mm"],
            ortho_metrics["overbite_mm"],
            ortho_metrics["midline_dev_mm"],
            ortho_metrics["anterior_contact_ratio"],
            ortho_metrics["posterior_contact_ratio"],
            ortho_metrics["left_contact_force"],
            ortho_metrics["right_contact_force"],
            ortho_metrics["working_side_interference"],
            ortho_metrics["nonworking_side_interference"],
            ortho_metrics["anterior_openbite_fraction"],
            ortho_metrics["posterior_crossbite_count"],
            ortho_metrics["scissors_bite_count"]
        ]
        
        return np.concatenate([
            rot, 
            trans, 
            m_vals
        ]).astype(np.float32)

    def render(self):
        pass


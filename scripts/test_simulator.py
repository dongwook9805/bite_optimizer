import sys
import os
import numpy as np
import trimesh

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from simulator import BiteSimulator
from utils import reward_function

def create_dummy_meshes():
    """Create two simple meshes: a plane (maxilla) and a sphere (mandible)."""
    # Maxilla: Flat plane at z=5
    maxilla = trimesh.creation.box(extents=[10, 10, 1])
    maxilla.apply_translation([0, 0, 5])
    
    # Mandible: Sphere at z=0 (initial)
    mandible = trimesh.creation.icosphere(radius=1.0)
    mandible.apply_translation([0, 0, 0])
    
    # Get absolute path to assets dir relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    assets_dir = os.path.join(project_root, 'assets')
    
    os.makedirs(assets_dir, exist_ok=True)
    
    maxilla_path = os.path.join(assets_dir, 'dummy_maxilla.stl')
    mandible_path = os.path.join(assets_dir, 'dummy_mandible.stl')
    
    maxilla.export(maxilla_path)
    mandible.export(mandible_path)
    print(f"Created dummy meshes in {assets_dir}")
    return maxilla_path, mandible_path

def test_simulator():
    maxilla_path, mandible_path = create_dummy_meshes()
    
    sim = BiteSimulator(maxilla_path, mandible_path)
    
    print("Initial State Metrics:")
    metrics = sim.get_metrics()
    # Should be far apart
    # dist roughly 5 - 0.5 - 1 = 3.5? 
    # Maxilla z=5 (center), thickness 1 -> bottom face z=4.5
    # Mandible z=0 (center), radius 1 -> top face z=1.0
    # Expected distance ~ 3.5
    print(f"Mean dist: {np.mean(metrics['distances']):.3f}")
    
    # Move mandible up by 3mm
    print("\nMoving UP by 3.0mm...")
    sim.apply_transform(np.array([0,0,0]), np.array([0,0,3.0]))
    
    metrics = sim.get_metrics()
    # Now top of sphere z=1+3=4. Bottom of box z=4.5. Dist should be ~0.5
    min_dist = np.min(metrics['distances'])
    print(f"Min dist after move: {min_dist:.3f}")
    
    # Calculate reward
    rew = reward_function(metrics['distances'], metrics['points'])
    print(f"Reward: {rew:.3f}")
    
    # Move up another 1.0mm -> z=5. Penetration of 0.5mm?
    print("\nMoving UP by 1.0mm (Penetration)...")
    sim.apply_transform(np.array([0,0,0]), np.array([0,0,1.0]))
    metrics = sim.get_metrics()
    
    # Check if signed distance became negative (if our logic in simulator is simplistic, it might just be small positive if we didn't handle sign well)
    # The current simulator logic uses 'on_surface' which returns unsigned distance usually unless we use signed_distance.
    # In my simulator code I used logic with normals to try to detect sign.
    # Let's see if it works for this case.
    # Sphere center z=4. Radius 1. Top z=5.
    # Box bottom z=4.5. 
    # Overlap range z=[4.5, 5].
    
    min_dist = np.min(metrics['distances']) # This might be negative if logic works
    print(f"Min dist (should be neg): {min_dist:.3f}")
    
    rew = reward_function(metrics['distances'], metrics['points'])
    print(f"Reward (should be low/neg): {rew:.3f}")

if __name__ == "__main__":
    test_simulator()

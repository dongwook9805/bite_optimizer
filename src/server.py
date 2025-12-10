from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import os
import sys
import numpy as np
import trimesh

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simulator import BiteSimulator
from utils import orthodontic_occlusion_reward

app = FastAPI()

# Initialize Simulator
# We'll use the dummy assets for now or look for real ones
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
MAXILLA_PATH = os.path.join(ASSETS_DIR, 'dummy_maxilla.stl')
MANDIBLE_PATH = os.path.join(ASSETS_DIR, 'dummy_mandible.stl')

if not os.path.exists(MAXILLA_PATH):
    # Fallback or error, for now let's hope they exist from previous steps
    print(f"Warning: Assets not found at {ASSETS_DIR}")

simulator = BiteSimulator(MAXILLA_PATH, MANDIBLE_PATH)

# Serve static files (HTML, JS, Assets)
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

class Action(BaseModel):
    dRx: float
    dRy: float
    dRz: float
    dTx: float
    dTy: float
    dTz: float

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(WEB_DIR, 'index.html'))

@app.get("/api/state")
async def get_state():
    """
    Return current mandible transformation matrix and metrics.
    """
    matrix = simulator.transform_matrix.tolist()
    metrics = simulator.get_metrics()
    
    ortho_metrics = simulator.get_orthodontic_metrics()
    reward = orthodontic_occlusion_reward(**ortho_metrics)
    
    # Merge basic metrics and ortho metrics for display
    display_metrics = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in metrics.items() if k != 'points'}
    display_metrics.update(ortho_metrics)
    
    return {
        "matrix": matrix,
        "reward": reward,
        "metrics": display_metrics
    }

@app.post("/api/reset")
async def reset_sim():
    simulator.reset()
    
    # Apply random perturbation to make it "misaligned"
    # Rotate up to +/- 10 degrees, Translate up to +/- 5mm
    rx, ry, rz = np.random.uniform(-10, 10, 3)
    tx, ty, tz = np.random.uniform(-5, 5, 3)
    
    simulator.apply_transform(np.array([rx, ry, rz]), np.array([tx, ty, tz]))
    print(f"Reset to randomized state: R=[{rx:.2f}, {ry:.2f}, {rz:.2f}], T=[{tx:.2f}, {ty:.2f}, {tz:.2f}]")
    
    return {"status": "reset_randomized"}

@app.post("/api/icp")
async def run_icp():
    simulator.rough_align_icp()
    
    metrics = simulator.get_metrics()
    ortho_metrics = simulator.get_orthodontic_metrics()
    reward = orthodontic_occlusion_reward(**ortho_metrics)
    
    display_metrics = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in metrics.items() if k != 'points'}
    display_metrics.update(ortho_metrics)
    
    return {
        "status": "icp_done",
        "reward": reward,
        "metrics": display_metrics
    }

@app.post("/api/step")
async def step_sim(action: Action):
    # Scale? The Env did scaling. Here we might take raw deltas.
    # Let's assume raw deltas.
    
    delta_r = np.array([action.dRx, action.dRy, action.dRz])
    delta_t = np.array([action.dTx, action.dTy, action.dTz])
    
    simulator.apply_transform(delta_r, delta_t)
    
    metrics = simulator.get_metrics()
    ortho_metrics = simulator.get_orthodontic_metrics()
    reward = orthodontic_occlusion_reward(**ortho_metrics)
    
    display_metrics = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in metrics.items() if k != 'points'}
    display_metrics.update(ortho_metrics)
    
    return {
        "reward": reward,
        "metrics": display_metrics
    }

    return {
        "reward": reward,
        "metrics": display_metrics
    }

# RL Model Loading
from stable_baselines3 import PPO
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'ppo_bite.zip')
rl_model = None

if os.path.exists(model_path):
    try:
        rl_model = PPO.load(model_path)
        print(f"Loaded RL model from {model_path}")
    except Exception as e:
        print(f"Failed to load RL model: {e}")
else:
    print(f"RL model not found at {model_path}")

@app.post("/api/rl_step")
async def rl_step_sim():
    if rl_model is None:
        return {"error": "RL model not loaded"}
        
    # Construct Observation (Must match BiteEnv._get_obs)
    # [Rot(3), Trans(3), Metrics(12)] = 18 dims
    
    trans = simulator.current_translation
    rot = np.zeros(3) # Dummy rotation as in Env
    
    ortho_metrics = simulator.get_orthodontic_metrics()
    
    # Order matches bite_env.py
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
    
    obs = np.concatenate([rot, trans, m_vals]).astype(np.float32)
    
    # Predict
    action, _ = rl_model.predict(obs, deterministic=True)
    
    # Scale Action (Must match BiteEnv parameters)
    # Env max_rot=0.5, max_trans=0.1
    MAX_ROT = 0.5
    MAX_TRANS = 0.1
    
    delta_r = action[:3] * MAX_ROT
    delta_t = action[3:] * MAX_TRANS
    
    # Apply
    simulator.apply_transform(delta_r, delta_t)
    
    # Return new state
    metrics = simulator.get_metrics()
    ortho_metrics = simulator.get_orthodontic_metrics()
    reward = orthodontic_occlusion_reward(**ortho_metrics)
    
    display_metrics = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in metrics.items() if k != 'points'}
    display_metrics.update(ortho_metrics)
    
    return {
        "status": "rl_step_done",
        "reward": reward,
        "metrics": display_metrics
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

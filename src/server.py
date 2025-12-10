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
from utils import reward_function

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
    reward = reward_function(metrics['distances'], metrics['points'])
    
    return {
        "matrix": matrix,
        "reward": reward,
        "metrics": {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in metrics.items() if k != 'points'}
    }

@app.post("/api/reset")
async def reset_sim():
    simulator.reset()
    return {"status": "reset"}

@app.post("/api/step")
async def step_sim(action: Action):
    # Scale? The Env did scaling. Here we might take raw deltas.
    # Let's assume raw deltas.
    
    delta_r = np.array([action.dRx, action.dRy, action.dRz])
    delta_t = np.array([action.dTx, action.dTy, action.dTz])
    
    simulator.apply_transform(delta_r, delta_t)
    
    metrics = simulator.get_metrics()
    reward = reward_function(metrics['distances'], metrics['points'])
    
    return {
        "reward": reward,
        "metrics": {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in metrics.items() if k != 'points'}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

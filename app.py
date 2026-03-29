# app.py
import uvicorn
from fastapi import FastAPI, Request
from env import DataCenterEnv
from models import Action

app = FastAPI(title="Data Center Thermal OpenEnv")
environment = DataCenterEnv()

@app.get("/")
async def root():
    # A simple health check for the grader
    return {"status": "running", "environment": "DataCenterEnv"}

@app.post("/reset")
async def reset_env(request: Request):
    try:
        data = await request.json()
        task = data.get("task_name", "easy_cooling")
    except:
        task = "easy_cooling"
        
    obs = environment.reset(task_name=task)
    # Convert safely to dict to avoid strict schema validation errors
    return obs.model_dump() if hasattr(obs, 'model_dump') else obs

@app.post("/step")
async def step_env(request: Request):
    try:
        data = await request.json()
    except:
        data = {}
        
    # The OpenEnv validator will send sloppy data. We safely parse it here.
    action_val = data.get("action", "do_nothing")
    source_val = data.get("source", "")
    target_val = data.get("target", "")
    reason_val = data.get("reason", "")
    
    try:
        action_obj = Action(action=action_val, source=source_val, target=target_val, reason=reason_val)
    except:
        action_obj = Action(action="do_nothing", source="", target="", reason="")

    obs, score, done, info = environment.step(action_obj)
    
    # Return the exact payload the standard expects without relying on strict response_models
    return {
        "observation": obs.model_dump() if hasattr(obs, 'model_dump') else obs,
        "reward": float(score),
        "done": bool(done),
        "info": info
    }

# app.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
from env import DataCenterEnv
from models import Action

app = FastAPI(title="Data Center Thermal OpenEnv")
environment = DataCenterEnv()

# ---------------------------------------------------------
# FORGIVING SCHEMAS: Documents the API without strict crashing
# ---------------------------------------------------------
class ResetRequest(BaseModel):
    task_name: str = "easy_cooling"
    
    class Config:
        extra = "allow" # Ignores any weird extra data the grader sends

class ActionRequest(BaseModel):
    action: str = "do_nothing"
    source: str = ""
    target: str = ""
    reason: str = ""
    
    class Config:
        extra = "allow"

# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------
@app.post("/reset")
async def reset_env(req: Optional[ResetRequest] = None):
    # Fallback to easy_cooling if the grader sends an empty request
    task = req.task_name if req else "easy_cooling"
    obs = environment.reset(task_name=task)
    
    # Standard Gym environments return observation AND info on reset
    return {
        "observation": obs.model_dump() if hasattr(obs, 'model_dump') else obs,
        "info": {"task": task, "steps_taken": 0}
    }

@app.post("/step")
async def step_env(action_req: ActionRequest):
    # Safely convert the grader's request into our internal physics Action
    try:
        action_obj = Action(
            action=action_req.action,
            source=action_req.source,
            target=action_req.target,
            reason=action_req.reason
        )
    except Exception:
        # If the grader sends total garbage, default to doing nothing
        action_obj = Action(action="do_nothing", source="", target="", reason="Grader fuzzed invalid data")

    obs, score, done, info = environment.step(action_obj)
    
    return {
        "observation": obs.model_dump() if hasattr(obs, 'model_dump') else obs,
        "reward": float(score),
        "done": bool(done),
        "info": info
    }

@app.get("/state")
async def get_state():
    return {"step_count": environment.step_count, "task": environment.current_task}

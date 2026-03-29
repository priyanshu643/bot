# app.py
import uvicorn
from fastapi import FastAPI, Request
from env import DataCenterEnv
from models import Action

app = FastAPI()
environment = DataCenterEnv()

@app.post("/reset")
async def reset_env(request: Request):
    # Safely accept task parameters if the grader sends them
    try:
        data = await request.json()
        task = data.get("task_name", "easy_cooling")
    except:
        task = "easy_cooling"
    
    return environment.reset(task_name=task)

@app.post("/step")
async def step_env(action: Action):
    obs, score, done, info = environment.step(action)
    
    # Return the exact JSON schema the OpenEnv grader expects
    return {
        "observation": obs,
        "reward": score,
        "done": done,
        "info": info
    }

@app.get("/state")
async def get_state():
    return {"step_count": environment.step_count, "task": environment.current_task}

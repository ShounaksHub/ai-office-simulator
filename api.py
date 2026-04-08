from fastapi import FastAPI
from pydantic import BaseModel
from env import OfficeEnv, Action

app = FastAPI()

env = None

class StepRequest(BaseModel):
    action_type: str
    email_index: int
    response: str = ""

@app.post("/reset")
def reset(task: str = "easy"):
    global env
    env = OfficeEnv(task_type=task)
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step(req: StepRequest):
    global env
    action = Action(**req.dict())
    obs, reward, done, _ = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done
    }

@app.get("/state")
def state():
    global env
    return {"state": [e.model_dump() for e in env.state()]}
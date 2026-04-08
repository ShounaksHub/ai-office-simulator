from fastapi import FastAPI
from pydantic import BaseModel
from env import OfficeEnv, Action
import uvicorn

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

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
    action = Action(**req.model_dump())
    obs, reward, done, _ = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done
    }

@app.get("/state")
def state():
    return {"state": [e.model_dump() for e in env.state()]}

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

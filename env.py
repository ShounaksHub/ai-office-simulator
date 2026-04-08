from pydantic import BaseModel
from typing import List, Tuple, Dict, Any

# ----------------------
# Data Models
# ----------------------
class Email(BaseModel):
    sender: str
    subject: str
    content: str
    priority: int


class Observation(BaseModel):
    emails: List[Email]
    time: int
    task_type: str


class Action(BaseModel):
    action_type: str
    email_index: int
    response: str = ""


# ----------------------
# Environment
# ----------------------
class OfficeEnv:
    def __init__(self, task_type="easy"):
        self.task_type = task_type
        self.time = 0
        self.handled = set()
        self.state_data = []

    def reset(self) -> Observation:
        self.time = 0
        self.handled = set()

        if self.task_type == "easy":
            self.state_data = [
                Email(sender="boss", subject="Meeting", content="Join ASAP", priority=5),
                Email(sender="newsletter", subject="News", content="FYI", priority=1),
            ]

        elif self.task_type == "medium":
            self.state_data = [
                Email(sender="client", subject="Update", content="Need status", priority=4),
                Email(sender="team", subject="Docs", content="Review this", priority=2),
            ]

        else:  # hard
            self.state_data = [
                Email(sender="boss", subject="Critical", content="Client escalation", priority=5),
                Email(sender="client", subject="Delay", content="Why delay?", priority=4),
                Email(sender="newsletter", subject="Promo", content="Sale!", priority=1),
            ]

        return self._get_obs()

    def state(self):
        return self.state_data

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        reward = 0.0
        done = False

        if action.email_index >= len(self.state_data):
            return self._get_obs(), -1.0, False, {"error": "Invalid index"}

        email = self.state_data[action.email_index]
        key = (action.email_index, action.action_type)

        # Prevent repetition
        if key in self.handled:
            reward -= 1.0
        else:
            self.handled.add(key)

        # Task-specific reward
        if action.action_type == "reply":
            if email.priority >= 4:
                reward += 1.0
            else:
                reward -= 0.3

        elif action.action_type == "classify":
            reward += 0.5 if email.priority >= 3 else 0.2

        elif action.action_type == "prioritize":
            reward += email.priority / 5.0

        # Time penalty
        reward -= 0.1

        self.time += 1

        if self.time >= 5:
            done = True

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return Observation(
            emails=self.state_data,
            time=self.time,
            task_type=self.task_type
        )
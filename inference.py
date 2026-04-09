import os
import requests
from openai import OpenAI

# ----------- ENV VARIABLES -----------
API_BASE_URL = os.environ["API_BASE_URL"]  
API_KEY = os.environ["API_KEY"]

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# Your environment (HF Space API)
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

# ----------- OPENAI CLIENT -----------
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

BENCHMARK = "ai_office_simulator"
MAX_STEPS = 5


# ----------- DECISION FUNCTION -----------
def decide_action(observation):
    emails = observation["emails"]

    email_text = "\n".join([
        f"{i}: {e['subject']} (priority {e['priority']})"
        for i, e in enumerate(emails)
    ])

    prompt = f"""
You are an AI assistant.

Given these emails:
{email_text}

Choose the best action:
- reply
- classify
- prioritize

Return ONLY in format:
action_type,email_index
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20
    )

    output = response.choices[0].message.content.strip()

    try:
        action_type, email_index = output.split(",")
        email_index = int(email_index)
    except:
        action_type = "classify"
        email_index = 0

    return {
        "action_type": action_type,
        "email_index": email_index,
        "response": "Handled"
    }


# ----------- RUN TASK -----------
def run_task(task):
    rewards = []
    steps = 0

    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        # RESET ENV
        obs = requests.post(f"{ENV_BASE_URL}/reset", params={"task": task}).json()
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = decide_action(obs)

            # STEP ENV
            res = requests.post(f"{ENV_BASE_URL}/step", json=action).json()

            obs = res["observation"]
            reward = float(res["reward"])
            done = res["done"]

            rewards.append(reward)
            steps = step

            print(
                f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
                flush=True,
            )

            if done:
                break

        # ----------- SCORE NORMALIZATION -----------
        total_reward = sum(rewards)
        max_possible = MAX_STEPS * 1.0
        min_possible = MAX_STEPS * -1.0

        score = (total_reward - min_possible) / (max_possible - min_possible)
        score = max(0.0, min(1.0, score))

        success = score >= 0.3

    except Exception as e:
        success = False
        score = 0.0

    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ----------- MAIN -----------
if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)

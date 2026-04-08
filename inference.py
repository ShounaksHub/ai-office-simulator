import os
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "dummy")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

BENCHMARK = "ai_office_simulator"
MAX_STEPS = 5


def decide_action(observation):
    emails = observation["emails"]
    idx = max(range(len(emails)), key=lambda i: emails[i]["priority"])

    return {
        "action_type": "reply" if emails[idx]["priority"] >= 4 else "classify",
        "email_index": idx,
        "response": "Handling task"
    }


def run_task(task):
    rewards = []
    steps = 0

    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        obs = requests.post(f"{API_BASE_URL}/reset", params={"task": task}).json()
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = decide_action(obs)

            res = requests.post(f"{API_BASE_URL}/step", json=action).json()

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

        # Normalize score to [0,1]
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


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_task(task)
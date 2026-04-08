# AI Office Simulator (OpenEnv)

## Overview
This environment simulates real-world office workflows such as email handling, prioritization, and responses.
And this environment is fully deployable and reproducible via Hugging Face Spaces using containerized execution

## Tasks
- Easy: Email classification
- Medium: Email response generation
- Hard: Multi-task workflow management

## Action Space
- classify
- reply
- prioritize

## Observation Space
- List of emails (sender, subject, content, priority)
- Current time step
- Task type

## Reward System
- Positive reward for correct actions
- Penalty for wrong actions and repetition
- Time penalty for inefficiency

## Setup
```bash
docker build -t ai-office-env .
docker run ai-office-env

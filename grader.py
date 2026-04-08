def grade(actions_log):
    total_reward = sum([reward for _, reward in actions_log])

    max_possible = len(actions_log) * 1.0
    min_possible = len(actions_log) * -1.0

    # Normalize score between 0 and 1
    normalized = (total_reward - min_possible) / (max_possible - min_possible)

    return round(normalized, 3)
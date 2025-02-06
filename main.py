import numpy as np

# Define the decision matrix (rows = actions, columns = states)
payoff_matrix = np.array([
    [6, 0, 3],   # Action 1
    [10, 2, 0],   # Action 2
    [9, 0, 10],   # Action 3
    [6, 8, 2]    # Action 4
])

# Maximin Rule: Pick the action with the highest minimum payoff
min_payoffs = np.min(payoff_matrix, axis=1)
maximin_action = np.argmax(min_payoffs)

# Minimax Regret Rule: Compute regret matrix and pick the action with the lowest max regret
max_per_column = np.max(payoff_matrix, axis=0)
regret_matrix = max_per_column - payoff_matrix
max_regrets = np.max(regret_matrix, axis=1)
minimax_regret_action = np.argmin(max_regrets)

# Optimism-Pessimism Rule (𝛼 = 1/2)
alpha = 0.5
weighted_payoffs = alpha * np.max(payoff_matrix, axis=1) + (1 - alpha) * np.min(payoff_matrix, axis=1)
optimism_pessimism_action = np.argmax(weighted_payoffs)

# Principle of Insufficient Reason: Average payoffs across states, pick the best
average_payoffs = np.mean(payoff_matrix, axis=1)
principle_insufficient_reason_action = np.argmax(average_payoffs)

# Print results (actions are zero-indexed)
print(f"Maximin Action: {maximin_action + 1}")
print(f"Minimax Regret Action: {minimax_regret_action + 1}")
print(f"Optimism-Pessimism Action: {optimism_pessimism_action + 1}")
print(f"Principle of Insufficient Reason Action: {principle_insufficient_reason_action + 1}")

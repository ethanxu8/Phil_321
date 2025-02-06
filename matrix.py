import numpy as np

def maximin_choice(matrix):
    min_payoffs = np.min(matrix, axis=1)
    return np.argmax(min_payoffs)

def minimax_regret_choice(matrix):
    max_per_column = np.max(matrix, axis=0)
    regret_matrix = max_per_column - matrix
    max_regrets = np.max(regret_matrix, axis=1)
    return np.argmin(max_regrets)

def optimism_pessimism_choice(matrix, alpha=0.5):
    weighted_payoffs = alpha * np.max(matrix, axis=1) + (1 - alpha) * np.min(matrix, axis=1)
    return np.argmax(weighted_payoffs)

def principle_insufficient_reason_choice(matrix):
    average_payoffs = np.mean(matrix, axis=1)
    return np.argmax(average_payoffs)

def generate_valid_matrix():
    while True:
        # Generate a random 4x3 matrix with values between 0 and 10
        matrix = np.random.randint(0, 11, (4, 3))
        
        # Compute choices for each rule
        maximin_action = maximin_choice(matrix)
        minimax_regret_action = minimax_regret_choice(matrix)
        optimism_pessimism_action = optimism_pessimism_choice(matrix)
        principle_insufficient_reason_action = principle_insufficient_reason_choice(matrix)
        
        # Ensure all rules select different rows
        selected_actions = {maximin_action, minimax_regret_action, optimism_pessimism_action, principle_insufficient_reason_action}
        if len(selected_actions) == 4:
            return matrix

# Generate a valid matrix
valid_matrix = generate_valid_matrix()

# Print the matrix
print("Generated Decision Matrix:")
print(valid_matrix)

# Print which rule picks which row
print(f"Maximin Rule chooses row {maximin_choice(valid_matrix) + 1}")
print(f"Minimax Regret Rule chooses row {minimax_regret_choice(valid_matrix) + 1}")
print(f"Optimism-Pessimism Rule chooses row {optimism_pessimism_choice(valid_matrix) + 1}")
print(f"Principle of Insufficient Reason chooses row {principle_insufficient_reason_choice(valid_matrix) + 1}")

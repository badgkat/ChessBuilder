import numpy as np

# Load the training data file.
data = np.load("training_data.npz")
states = data['states']
policy_targets = data['policy_targets']
value_targets = data['value_targets']

print("States shape:", states.shape)           # e.g., (num_samples, 13, 8, 8)
print("Policy targets shape:", policy_targets.shape)  # e.g., (num_samples, 8513)
print("Value targets shape:", value_targets.shape)    # e.g., (num_samples, 1)

# Print summary statistics.
print("\nPolicy targets statistics:")
print("Min:", policy_targets.min())
print("Max:", policy_targets.max())
print("Mean:", policy_targets.mean())
print("Std:", policy_targets.std())

print("\nValue targets statistics:")
print("Unique values:", np.unique(value_targets))

# Inspect a single sample
sample_index = 0
sample_state = states[sample_index]
sample_policy = policy_targets[sample_index]
sample_value = value_targets[sample_index]

print("\nSample #0:")
print("Board state (channel-wise summary):")
for i, channel in enumerate(sample_state):
    print(f"Channel {i}: sum = {channel.sum()}")  # this gives a rough idea of piece counts

print("\nPolicy target vector (non-zero indices):")
nonzero_indices = np.nonzero(sample_policy)[0]
print("Non-zero indices:", nonzero_indices)
print("Values at non-zero indices:", sample_policy[nonzero_indices])
print("\nValue target:", sample_value)

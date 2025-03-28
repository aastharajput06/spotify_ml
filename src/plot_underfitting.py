import numpy as np
import matplotlib.pyplot as plt

# Hypothetical data for an underfitting model
data_sizes = np.linspace(100, 1000, 50)  # Training instances size
train_scores = 0.6 + 0.05 * (data_sizes - 100) / 900  # Linear increase from 0.6 to 0.65
val_scores = 0.58 + 0.05 * (data_sizes - 100) / 900  # Linear increase from 0.58 to 0.63

# Plot
plt.figure(figsize=(8, 6))
plt.plot(data_sizes, train_scores, label='Training Score', color='blue')
plt.plot(data_sizes, val_scores, label='Validation Score', color='orange')
plt.xlabel('Training Instances Size')
plt.ylabel('Score')
plt.title('Training and Validation Scores for an Underfitting Model')
plt.legend()
plt.grid(True)
plt.savefig('docs/underfitting_plot.png')
plt.close()
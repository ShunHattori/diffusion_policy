import json

import matplotlib.pyplot as plt
import numpy as np

# Load the data from the uploaded files
files = [
    'output/single180/eval_log.json',
    'output/single190/eval_log.json',
    'output/comp190-180-div2/eval_log.json'
]
xlabels = ['policy1', 'policy2', 'comp policy']
mean_scores = ['0.838', '0.854', '0.830']

# Extract the 50 trial success rates from the data
experiment_data = []
for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        trial_keys = [key for key in data.keys() if key.startswith('test/sim_max_reward') and not key.endswith('video')]
        success_rates = [data[key] for key in trial_keys]
        experiment_data.append(success_rates)

# Calculate mean and variance for each experiment
means = [np.mean(data) for data in experiment_data]
variances = [np.var(data) for data in experiment_data]

# 1. Plotting Histograms for each experiment
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
bins = np.linspace(0, 1, 21)  # 20 bins between 0 and 1

for i, data in enumerate(experiment_data):
    axes[i].hist(data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    axes[i].set_title(xlabels[i])
    axes[i].set_xlabel('Success Rate')
    axes[i].set_ylabel('Frequency')
    axes[i].set_xlim(0, 1)
    axes[i].set_ylim(0, 30)  # Adjust as necessary for visibility
    axes[i].text(0.05, 0.95, f"Mean: {means[i]:.3f}\nVariance: {variances[i]:.3f}", transform=axes[i].transAxes, ha='left', va='top')

plt.tight_layout()
plt.savefig('output/mean_score_fig.png', dpi=300)

# 4. Plotting Bar Chart with Error Bars
fig, ax = plt.subplots(figsize=(8, 6))
x_positions = np.arange(1, len(means) + 1)

# Error bars using standard deviation
std_devs = [np.std(data) for data in experiment_data]
ax.bar(x_positions, means, yerr=std_devs, capsize=5, color='lightblue', edgecolor='black', alpha=0.7)

# Add labels and titles
ax.set_xticks(x_positions)
ax.set_xticklabels(xlabels)
ax.set_ylim(0, 1.2)
ax.set_title('Mean Success Rate')
ax.set_ylabel('Success Rate')

plt.tight_layout()
plt.savefig('output/mean_score_fig2.png', dpi=300)

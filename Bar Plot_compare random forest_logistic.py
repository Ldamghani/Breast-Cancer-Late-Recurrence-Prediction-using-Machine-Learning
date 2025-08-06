import matplotlib.pyplot as plt
import numpy as np

# Data for Logistic Regression and Random Forest
models = ['Logistic Regression', 'Random Forest']
accuracy = [0.7625, 0.7047]
precision = [0.7734, 0.3302]
recall = [0.9680, 0.1920]
f1_score = [0.8598, 0.2384]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define the positions for each bar
x = np.arange(len(models))  # positions for the bars
width = 0.2  # width of each bar

# Create bars for each metric
ax.bar(x - width*1.5, accuracy, width, label='Accuracy', color='royalblue')
ax.bar(x - width/2, precision, width, label='Precision', color='darkorange')
ax.bar(x + width/2, recall, width, label='Recall', color='forestgreen')
ax.bar(x + width*1.5, f1_score, width, label='F1-Score', color='crimson')

# Add some text for labels, title, and custom x-axis tick labels
ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('Scores', fontsize=14)
ax.set_title('Comparison of Model Performance: Logistic Regression vs. Random Forest', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()

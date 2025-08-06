import matplotlib.pyplot as plt
import seaborn as sns

# Data (based on your previous filtered group)
labels = ["No Late Recurrence", "Late Recurrence"]
percentages = [81.47, 18.53]

# Set up the plot
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
ax = sns.barplot(x=labels, y=percentages, palette=["skyblue", "salmon"])

# Add text labels above bars
for i, v in enumerate(percentages):
    ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

# Set titles and labels
plt.title("Late Recurrence in Patients (Age > 50, ER+/PR+, HER2−)", fontsize=12)
plt.ylabel("Percentage (%)")
plt.ylim(0, 100)
plt.tight_layout()

# Show the plot
plt.show()

# Save the plot as a PNG image
plt.savefig("late_recurrence_barplot.png", dpi=300)

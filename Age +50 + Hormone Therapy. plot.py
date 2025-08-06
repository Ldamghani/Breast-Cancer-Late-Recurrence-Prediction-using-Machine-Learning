import matplotlib.pyplot as plt
import seaborn as sns

# Data
labels = ["No Hormone Therapy", "Yes Hormone Therapy"]
percentages = [15.79, 19.54]

# Plot setup
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
ax = sns.barplot(x=labels, y=percentages, palette=["gray", "green"])

# Add data labels above bars
for i, v in enumerate(percentages):
    ax.text(i, v + 0.7, f"{v:.2f}%", ha='center', fontweight='bold')

# Titles and labels
plt.title("Late Recurrence Rate by Hormone Therapy in Age > 50 (ER+/PR+/HER2−)", fontsize=11)
plt.ylabel("Late Recurrence (%)")
plt.ylim(0, 25)
plt.tight_layout()

# Save and Show
plt.savefig("late_recurrence_by_hormone_therapy.png", dpi=300)
plt.show()

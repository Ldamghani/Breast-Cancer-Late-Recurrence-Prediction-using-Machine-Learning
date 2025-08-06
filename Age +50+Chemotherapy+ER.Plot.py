import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define data manually from your previous output
data = {
    "ER Status": ["Negative", "Negative", "Positive", "Positive"],
    "Chemotherapy": ["No", "Yes", "No", "Yes"],
    "Late Recurrence (%)": [5.38, 9.84, 16.88, 25.42]
}

df_plot = pd.DataFrame(data)

# Plot setup
plt.figure(figsize=(7, 5))
sns.set(style="whitegrid")
ax = sns.barplot(
    x="Chemotherapy",
    y="Late Recurrence (%)",
    hue="ER Status",
    data=df_plot,
    palette="Set2"
)

# Annotate bars with values
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f%%', label_type='edge', fontsize=9)

# Titles and labels
plt.title("Late Recurrence by Chemotherapy and ER Status (Age > 50)", fontsize=12)
plt.ylim(0, 30)
plt.ylabel("Late Recurrence (%)")
plt.tight_layout()

# Save and show
plt.savefig("late_recurrence_by_chemo_er.png", dpi=300)
plt.show()

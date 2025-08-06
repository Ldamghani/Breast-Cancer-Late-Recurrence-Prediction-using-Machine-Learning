import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Load data from Step1
file_path = r"D:\Leila USA\Master Courses\breast cancer owrang\My published Breast Cancer Data Analysis\last\data\2, Breast Cancer METABRIC.xlsx"
df = pd.read_excel(file_path, sheet_name="Copy, Breast Cancer METABRIC")

# Create Data and clean miss data
df["Late_Recurrence"] = (
    (df["Relapse Free Status"] == "Recurred") &
    (df["Relapse Free Status (Months)"] >= 60)
).astype(int)

df_clean = df.dropna()

# Age division
df_clean["Age_Group"] = df_clean["Age at Diagnosis"].apply(
    lambda x: "Over 50" if x > 50 else "50 or younger"
)

# Select data type (Exept Late_Recurrence)
X = df_clean.select_dtypes(include=["number"]).drop(columns=["Late_Recurrence"])
y = df_clean["Late_Recurrence"]

# divide age Over 50 and below 50
X_young = X[df_clean["Age_Group"] == "50 or younger"]
y_young = y[df_clean["Age_Group"] == "50 or younger"]

X_old = X[df_clean["Age_Group"] == "Over 50"]
y_old = y[df_clean["Age_Group"] == "Over 50"]

#  create Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross function 10 foe each group
acc_young = cross_val_score(rf, X_young, y_young, cv=10, scoring="accuracy")
acc_old = cross_val_score(rf, X_old, y_old, cv=10, scoring="accuracy")

# Print Result
print("📊  Accuracy Random Forest:")
print(f"- Below 50>= {round(np.mean(acc_young)*100, 2)}%")
print(f"- Over 50<: {round(np.mean(acc_old)*100, 2)}%")
import matplotlib.pyplot as plt

plt.bar(["≤50", ">50"], [np.mean(acc_young)*100, np.mean(acc_old)*100], color=["skyblue", "salmon"])
plt.ylabel("Accuracy (%)")
plt.title("Random Forest Accuracy by Age Group")
plt.show()

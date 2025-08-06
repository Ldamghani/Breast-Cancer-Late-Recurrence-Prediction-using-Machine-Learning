import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Load Excel dataset
file_path = file_path = r"D:\Leila USA\Master Courses\breast cancer owrang\breast_cancer_project\2, Breast Cancer METABRIC.xlsx"


df = pd.read_excel(file_path, sheet_name="Copy, Breast Cancer METABRIC")

# Create Late Recurrence label
df["Late_Recurrence"] = (
    (df["Relapse Free Status"] == "Recurred") &
    (df["Relapse Free Status (Months)"] >= 60)
).astype(int)

# Clean dataset
df_clean = df.dropna()

# Filter subgroup: Age > 50, ER+, PR+, HER2−
subset = df_clean[
    (df_clean["Age at Diagnosis"] > 50) &
    (df_clean["ER Status"] == "Positive") &
    (df_clean["PR Status"] == "Positive") &
    (df_clean["HER2 Status"] == "Negative")
]

# Select numerical features
X = subset.select_dtypes(include=["number"]).drop(columns=["Late_Recurrence"])
y = subset["Late_Recurrence"]

# Define and evaluate Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
accuracy = cross_val_score(rf, X, y, cv=10, scoring="accuracy")

# Print average accuracy
print(f"Mean Accuracy (10-fold CV): {round(np.mean(accuracy) * 100, 2)}%")

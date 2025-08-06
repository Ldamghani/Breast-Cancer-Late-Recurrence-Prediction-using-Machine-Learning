# Cross_Validation.py

import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load Excel file
file_path = r"D:\Leila USA\Master Courses\breast cancer owrang\My published Breast Cancer Data Analysis\last\data\2, Breast Cancer METABRIC.xlsx"
df = pd.read_excel(file_path, sheet_name="Copy, Breast Cancer METABRIC")

# Create target column based on survival time (>= 60 months = late recurrence)
df["Late_Recurrence"] = df["Overall Survival (Months)"].apply(lambda x: 1 if x >= 60 else 0)

# Drop missing values in relevant columns
features = ["Age at Diagnosis", "Tumor Size", "ER Status", "PR Status", "HER2 Status", "Chemotherapy", "Hormone Therapy"]
df = df.dropna(subset=features + ["Overall Survival (Months)"])

# Encode categorical variables
df_encoded = df.copy()
for col in features:
    if df_encoded[col].dtype == "object":
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

# Prepare data
X = df_encoded[features]
y = df_encoded["Late_Recurrence"]

# Define model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print(f"Cross-validated Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")




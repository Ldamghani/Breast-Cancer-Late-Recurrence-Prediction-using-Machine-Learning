import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
file_path = r"D:\Leila USA\Master Courses\breast cancer owrang\My published Breast Cancer Data Analysis\last\data\2, Breast Cancer METABRIC.xlsx"
df = pd.read_excel(file_path, sheet_name="Copy, Breast Cancer METABRIC")

# Create target label
df["Late_Recurrence"] = df["Overall Survival (Months)"].apply(lambda x: 1 if x >= 60 else 0)

# Select features
features = ["Age at Diagnosis", "Tumor Size", "ER Status", "PR Status", "HER2 Status", "Chemotherapy", "Hormone Therapy"]
df = df.dropna(subset=features + ["Overall Survival (Months)"])

# Encode categorical features
df_encoded = df.copy()
for col in features:
    if df_encoded[col].dtype == "object":
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

X = df_encoded[features]
y = df_encoded["Late_Recurrence"]

# Split into training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Use SHAP's new API
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train, check_additivity=False)

# Check shape to confirm we’re accessing the right class
print("SHAP shape:", shap_values.values.shape)  # Should be (samples, features, 2)

# Plot SHAP summary for class 1
shap.summary_plot(shap_values.values[:, :, 1], X_train, plot_type="dot", max_display=10)








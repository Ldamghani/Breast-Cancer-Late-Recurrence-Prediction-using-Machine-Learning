import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = r"D:\Leila USA\Master Courses\breast cancer owrang\My published Breast Cancer Data Analysis\last\data\2, Breast Cancer METABRIC.xlsx"
df = pd.read_excel(file_path, sheet_name="Copy, Breast Cancer METABRIC")

# Create Late_Recurrence target variable based on 60 months
df["Late_Recurrence"] = df["Overall Survival (Months)"].apply(lambda x: 1 if x >= 60 else 0)

# Drop rows with missing values in required columns
features = ["Age at Diagnosis", "Tumor Size", "ER Status", "PR Status", "HER2 Status", "Chemotherapy", "Hormone Therapy"]
df = df.dropna(subset=features + ["Overall Survival (Months)"])

# Encode categorical features
df_encoded = df.copy()
for col in features:
    if df_encoded[col].dtype == "object":
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

# Prepare input and output
X = df_encoded[features]
y = df_encoded["Late_Recurrence"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importances
importances = model.feature_importances_

# Get feature names
features = X_train.columns

# Sort the feature importances in descending order
indices = np.argsort(importances)[::-1]

# Create a bar plot of feature importance
plt.figure(figsize=(10, 6))
plt.barh(features[indices], importances[indices], color='skyblue')
plt.xlabel('Importance Score', fontsize=14)
plt.title('Feature Importance for Predicting Late Recurrence', fontsize=16)
plt.show()

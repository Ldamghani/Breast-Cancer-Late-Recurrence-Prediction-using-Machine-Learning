import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the dataset
file_path = r"D:\Leila USA\Master Courses\breast cancer owrang\My published Breast Cancer Data Analysis\last\data\2, Breast Cancer METABRIC.xlsx"
df = pd.read_excel(file_path, sheet_name="Copy, Breast Cancer METABRIC")

# Create the target column (Late_Recurrence) based on 'Overall Survival (Months)'
df["Late_Recurrence"] = df["Overall Survival (Months)"].apply(lambda x: 1 if x >= 60 else 0)

# Drop rows with missing values in important columns
features = ["Age at Diagnosis", "Tumor Size", "ER Status", "PR Status", "HER2 Status", "Chemotherapy", "Hormone Therapy"]
df = df.dropna(subset=features + ["Overall Survival (Months)"])

# Encode categorical variables
df_encoded = df.copy()
for col in features:
    if df_encoded[col].dtype == "object":
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

# Prepare the features and target
X = df_encoded[features]
y = df_encoded["Late_Recurrence"]

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Set up 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation and calculate accuracy for each fold
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

# Plot the accuracy for each fold
plt.figure(figsize=(8, 6))
plt.plot(np.arange(1, 11), scores, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)
plt.title('Cross-Validation Accuracy for Random Forest', fontsize=16)
plt.xlabel('Fold', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(True)

# Set x and y axis ticks
plt.xticks(np.arange(1, 11))  # Set ticks for each fold
plt.yticks(np.arange(0.65, 0.75, 0.01))  # Set accuracy range

# Display the plot
plt.show()

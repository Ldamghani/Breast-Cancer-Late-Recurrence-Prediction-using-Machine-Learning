# logistic_regression_baseline.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# Set up cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000, solver='liblinear')  # Good for small/medium datasets

# Lists to store results
accuracy = []
precision = []
recall = []
f1 = []

# Perform 10-fold cross-validation
for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy.append(accuracy_score(y_test, y_pred))
    precision.append(precision_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))
    f1.append(f1_score(y_test, y_pred))

# Print average metrics
print("📊 Logistic Regression Performance (10-fold Cross-Validation):")
print(f"Accuracy:  {np.mean(accuracy):.4f}")
print(f"Precision: {np.mean(precision):.4f}")
print(f"Recall:    {np.mean(recall):.4f}")
print(f"F1 Score:  {np.mean(f1):.4f}")

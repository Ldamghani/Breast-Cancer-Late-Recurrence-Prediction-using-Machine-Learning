from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# Define 10-fold StratifiedKFold
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists for precision and recall
precision = []
recall = []

# Perform 10-fold cross-validation and calculate precision and recall for each fold
for train_index, test_index in cv.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    precision.append(precision_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred))

# Plot Precision and Recall for each fold
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, 11), precision, marker='o', label='Precision', color='r', linestyle='-', linewidth=2, markersize=8)
plt.plot(np.arange(1, 11), recall, marker='o', label='Recall', color='g', linestyle='-', linewidth=2, markersize=8)
plt.title('Precision and Recall for Each Fold', fontsize=16)
plt.xlabel('Fold', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.legend(loc='upper left')  # Display legend in the top left corner
plt.grid(True)

# Set ticks for each fold on x-axis
plt.xticks(np.arange(1, 11))

# Display the plot
plt.show()

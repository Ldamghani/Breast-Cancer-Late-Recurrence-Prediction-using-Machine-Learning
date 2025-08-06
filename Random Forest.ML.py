import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Step 1: Load the Excel file
file_path = r"D:\Leila USA\Master Courses\breast cancer owrang\My published Breast Cancer Data Analysis\last\data\2, Breast Cancer METABRIC.xlsx"
df = pd.read_excel(file_path, sheet_name="Copy, Breast Cancer METABRIC")

# Step 2: Create Late Recurrence label
df["Late_Recurrence"] = (
    (df["Relapse Free Status"] == "Recurred") &
    (df["Relapse Free Status (Months)"] >= 60)
).astype(int)

# Step 3: Select existing features only
features = [
    "Age at Diagnosis", "ER Status", "PR Status", "HER2 Status",
    "Chemotherapy", "Hormone Therapy", "Tumor Size"
]

df_model = df[features + ["Late_Recurrence"]].dropna()

# Step 4: Encode categorical columns
df_encoded = df_model.copy()
for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

# Step 5: Train Random Forest
X = df_encoded.drop("Late_Recurrence", axis=1)
y = df_encoded["Late_Recurrence"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Get feature importance
importances = model.feature_importances_
features = X.columns
importance_df = pd.Series(importances, index=features).sort_values(ascending=False)

# Step 7: Plot feature importance
plt.figure(figsize=(8, 5))
importance_df.plot(kind='barh', color="teal")
plt.title("Feature Importance for Predicting Late Recurrence")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance_late_recurrence.png", dpi=300)
plt.show()

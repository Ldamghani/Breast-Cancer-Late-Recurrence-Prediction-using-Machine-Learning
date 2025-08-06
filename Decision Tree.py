import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load Excel data
file_path = r"D:\Leila USA\Master Courses\breast cancer owrang\My published Breast Cancer Data Analysis\last\data\2, Breast Cancer METABRIC.xlsx"
df = pd.read_excel(file_path, sheet_name="Copy, Breast Cancer METABRIC")

# Create Late Recurrence target variable
df["Late_Recurrence"] = (
    (df["Relapse Free Status"] == "Recurred") &
    (df["Relapse Free Status (Months)"] >= 60)
).astype(int)

# Select reliable features
features = [
    "Age at Diagnosis", "ER Status", "PR Status", "HER2 Status",
    "Chemotherapy", "Hormone Therapy", "Tumor Size"
]
df_model = df[features + ["Late_Recurrence"]].dropna()

# Encode categorical features
df_encoded = df_model.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

# Define X and y
X = df_encoded.drop("Late_Recurrence", axis=1)
y = df_encoded["Late_Recurrence"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

# Predict
y_pred = dt_model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot the decision tree
plt.figure(figsize=(18, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=["No Late Recurrence", "Late Recurrence"],
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree for Late Recurrence Prediction")
plt.show()

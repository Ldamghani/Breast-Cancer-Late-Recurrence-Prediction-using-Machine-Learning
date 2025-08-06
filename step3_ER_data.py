import pandas as pd

# Load the METABRIC Excel file
file_path = r"D:\Leila USA\Master Courses\breast cancer owrang\My published Breast Cancer Data Analysis\last\data\2, Breast Cancer METABRIC.xlsx"
df = pd.read_excel(file_path, sheet_name="Copy, Breast Cancer METABRIC")

# Create Late Recurrence label
df["Late_Recurrence"] = (
    (df["Relapse Free Status"] == "Recurred") &
    (df["Relapse Free Status (Months)"] >= 60)
).astype(int)

# Clean data: remove rows with missing values
df_clean = df.dropna()

# Filter patients: Age > 50, ER+, PR+, HER2-
subset = df_clean[
    (df_clean["Age at Diagnosis"] > 50) &
    (df_clean["ER Status"] == "Positive") &
    (df_clean["PR Status"] == "Positive") &
    (df_clean["HER2 Status"] == "Negative")
]

# Show total number of patients in this subgroup
print("Total patients in selected subgroup:", subset.shape[0])

# Count Late Recurrence cases (1 = yes, 0 = no)
late_counts = subset["Late_Recurrence"].value_counts()
late_percent = late_counts / subset.shape[0] * 100

# Print summary
print("\nLate Recurrence distribution in this subgroup:")
print(late_counts)
print("\nPercentage:")
print(late_percent.round(2))


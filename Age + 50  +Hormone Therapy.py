import pandas as pd

# Step 1: Load Excel data
file_path = r"D:\Leila USA\Master Courses\breast cancer owrang\My published Breast Cancer Data Analysis\last\data\2, Breast Cancer METABRIC.xlsx"
df = pd.read_excel(file_path, sheet_name="Copy, Breast Cancer METABRIC")

# Step 2: Create Late Recurrence column
df["Late_Recurrence"] = (
    (df["Relapse Free Status"] == "Recurred") &
    (df["Relapse Free Status (Months)"] >= 60)
).astype(i)

# Step 3: Clean data
df_clean = df.dropna()

# Step 4: Filter by Age > 50 and hormone-sensitive profile
subset = df_clean[
    (df_clean["Age at Diagnosis"] > 50) &
    (df_clean["ER Status"] == "Positive") &
    (df_clean["PR Status"] == "Positive") &
    (df_clean["HER2 Status"] == "Negative")
]

# Step 5: Group by Hormone Therapy and Late Recurrence
summary = subset.groupby("Hormone Therapy")["Late_Recurrence"].value_counts().unstack().fillna(0)
summary["Total"] = summary[0] + summary[1]
summary["Late_Rec_Percent"] = (summary[1] / summary["Total"]) * 100

# Step 6: Print result
print(summary[["Total", 1, "Late_Rec_Percent"]].round(2))



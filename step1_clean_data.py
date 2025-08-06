import pandas as pd

# address excel file
file_path = r"D:\Leila USA\Master Courses\breast cancer owrang\My published Breast Cancer Data Analysis\last\data\2, Breast Cancer METABRIC.xlsx"


# load sheet orginal
df = pd.read_excel(file_path, sheet_name="Copy, Breast Cancer METABRIC")

#Creat late_Recurrence
df["Late_Recurrence"] = (
    (df["Relapse Free Status"] == "Recurred") &
    (df["Relapse Free Status (Months)"] >= 60)
).astype(int)

# Remove and clean data
df_clean = df.dropna()

# Print data
print("view data:", df_clean.shape)
print("tozi Late_Recurrence:")
print(df_clean["Late_Recurrence"].value_counts())

import pandas as pd
import kagglehub
import os

# Download dataset
path = kagglehub.dataset_download(
    "fedesoriano/heart-failure-prediction"
)

# Load dataset
csv_file = os.path.join(path, "heart.csv")
df = pd.read_csv(csv_file)

print("===== BEFORE CLEANING =====")
print("Dataset Shape:", df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\n===== AFTER CLEANING =====")
print("Dataset Shape:", df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows After Cleaning:")
print(df.duplicated().sum())

# Save cleaned dataset
df.to_csv("heart_cleaned.csv", index=False)

print("\nCleaned dataset saved as heart_cleaned.csv")

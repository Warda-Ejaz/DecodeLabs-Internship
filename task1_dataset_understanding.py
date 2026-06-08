import pandas as pd
import kagglehub
import os

print("Downloading dataset...")

# Download dataset
path = kagglehub.dataset_download(
    "fedesoriano/heart-failure-prediction"
)

print("Dataset downloaded successfully!")
print("Dataset path:", path)

# Load CSV file
csv_file = os.path.join(path, "heart.csv")
df = pd.read_csv(csv_file)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== DATASET INFO =====")
df.info()

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())
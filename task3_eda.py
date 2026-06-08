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

print("===== EXPLORATORY DATA ANALYSIS =====")

# Basic Statistics
print("\nStatistical Summary:")
print(df.describe())

# Mean
print("\nMean Values:")
print(df.mean(numeric_only=True))

# Median
print("\nMedian Values:")
print(df.median(numeric_only=True))

# Mode
print("\nMode Values:")
print(df.mode().iloc[0])

# Correlation Matrix
print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# Heart Disease Distribution
print("\nHeart Disease Counts:")
print(df['HeartDisease'].value_counts())

# Gender Distribution
print("\nGender Counts:")
print(df['Sex'].value_counts())

# Chest Pain Type Distribution
print("\nChest Pain Type Counts:")
print(df['ChestPainType'].value_counts())

# Average Age
print("\nAverage Age:")
print(df['Age'].mean())

# Maximum Heart Rate
print("\nAverage Max Heart Rate:")
print(df['MaxHR'].mean())

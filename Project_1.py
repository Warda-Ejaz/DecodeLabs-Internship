# Project 1: Advanced EDA & Feature Engineering

import pandas as pd
import numpy as np

df = pd.read_csv("student_performance.csv")

print("Original Dataset")
print(df)


print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

df["Study_Hours"] = df["Study_Hours"].fillna(df["Study_Hours"].mean())
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].mean())
df["Assignments"] = df["Assignments"].fillna(df["Assignments"].mean())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


Q1 = df["Study_Hours"].quantile(0.25)
Q3 = df["Study_Hours"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("\nLower Limit:", lower_limit)
print("Upper Limit:", upper_limit)


df["Study_Hours"] = np.where(
    df["Study_Hours"] > upper_limit,
    upper_limit,
    df["Study_Hours"]
)

df["Study_Hours"] = np.where(
    df["Study_Hours"] < lower_limit,
    lower_limit,
    df["Study_Hours"]
)

df["Study_Attendance"] = df["Study_Hours"] * df["Attendance"]

df["Assignment_Rate"] = df["Assignments"] / 10


df["Performance_Index"] = (
    (df["Study_Hours"] * 0.4)
    + (df["Attendance"] * 0.3)
    + (df["Assignments"] * 0.3)
)


print("\nCleaned Dataset")
print(df)


df.to_csv("cleaned_student_performance.csv", index=False)

print("\nProject Completed Successfully!")
print("Cleaned file saved as cleaned_student_performance.csv")

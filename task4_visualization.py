import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
import os

# Download dataset
path = kagglehub.dataset_download(
    "fedesoriano/heart-failure-prediction"
)

# Load dataset
csv_file = os.path.join(path, "heart.csv")
df = pd.read_csv(csv_file)

# -----------------------------
# 1. Bar Chart
# -----------------------------
df['HeartDisease'].value_counts().plot(kind='bar')
plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# -----------------------------
# 2. Pie Chart
# -----------------------------
df['Sex'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%'
)
plt.title("Gender Distribution")
plt.ylabel("")
plt.show()

# -----------------------------
# 3. Histogram
# -----------------------------
plt.hist(df['Age'], bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# 4. Scatter Plot
# -----------------------------
plt.scatter(df['Age'], df['MaxHR'])
plt.title("Age vs Maximum Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.show()

# -----------------------------
# 5. Box Plot
# -----------------------------
plt.boxplot(df['Cholesterol'])
plt.title("Cholesterol Distribution")
plt.ylabel("Cholesterol")
plt.show()

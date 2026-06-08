import pandas as pd
import kagglehub
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Download dataset
path = kagglehub.dataset_download(
    "fedesoriano/heart-failure-prediction"
)

# Load dataset
csv_file = os.path.join(path, "heart.csv")
df = pd.read_csv(csv_file)

# Convert categorical columns into numbers
label_encoder = LabelEncoder()

for column in df.select_dtypes(include='object').columns:
    df[column] = label_encoder.fit_transform(df[column])

# Features and Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = LogisticRegression(max_iter=1000)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("===== MODEL RESULTS =====")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

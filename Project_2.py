import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE


df = pd.read_csv("fraud_dataset.csv")

print("Dataset:")
print(df.head())

print("\nClass Distribution:")
print(df["Fraud"].value_counts())


X = df[["Transaction_Amount", "Transaction_Time"]]
y = df["Fraud"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

smote = SMOTE(
    random_state=42,
    k_neighbors=1
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())


lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_smote, y_train_smote)

lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:, 1]

print("\n===== Logistic Regression =====")

print("Precision:",
      precision_score(y_test, lr_pred, zero_division=0))

print("Recall:",
      recall_score(y_test, lr_pred, zero_division=0))

print("ROC-AUC:",
      roc_auc_score(y_test, lr_prob))


rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train_smote, y_train_smote)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n===== Random Forest =====")

print("Precision:",
      precision_score(y_test, rf_pred, zero_division=0))

print("Recall:",
      recall_score(y_test, rf_pred, zero_division=0))

print("ROC-AUC:",
      roc_auc_score(y_test, rf_prob))

print("\nProject Completed Successfully!")

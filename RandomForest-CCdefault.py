import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("default of credit card clients.csv")
df.head()

print(df.shape)
print(df.columns)
df.describe().T

for i in range(1, 7):
    df[f'utilization_{i}'] = df[f'BILL_AMT{i}'] / df['LIMIT_BAL']

# Average utilization
df['utilization_mean'] = df[[f'utilization_{i}' for i in range(1, 7)]].mean(axis=1)

# Replace infinities (if any LIMIT_BAL=0) and NaNs
df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
df.fillna(0, inplace=True)

df.head()

X = df.drop(columns = ["ID", "default payment next month"])
y = df["default payment next month"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 1, stratify = y
)

rf = RandomForestClassifier(
    n_estimators = 500,
    max_depth = None,
    random_state=1,
    class_weight = {0: 1, 1: 3.52}
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

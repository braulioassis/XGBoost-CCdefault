from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

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

# Create pipeline: scaling + logistic regression
log_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        max_iter = 500,
        class_weight = {0: 1, 1: 3.52},
        solver = 'liblinear',
        random_state = 1
    ))
])

log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1]

# Metrics
print("Logistic Regression Results:")
print("AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

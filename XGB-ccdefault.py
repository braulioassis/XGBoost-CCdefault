import pandas as pd

df = pd.read_csv("default of credit card clients.csv")
df.head()

print(df.shape)
print(df.columns)
df.describe().T

X = df.drop(columns = ["ID", "default payment next month"])
y = df["default payment next month"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 1, stratify = y
)

import xgboost as xgb

m1 = xgb.XGBClassifier(
    n_estimators = 500,
    learning_rate = 0.05,
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.8,
    eval_metric = 'logloss',
    random_state = 1,
)

m1.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
y_pred = m1.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

pos_weight = (y == 0).sum() / (y == 1).sum()

m2 = xgb.XGBClassifier(
    n_estimators = 500,
    learning_rate = 0.05,
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = pos_weight,
    eval_metric = 'logloss',
    random_state = 1
)

m2.fit(X_train, y_train)
y_pred = m2.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
xgb.plot_importance(m2, max_num_features = 10)
plt.show()

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

y_pred_proba = m2.predict_proba(X_test)[:, 1]
prec, rec, thresh = precision_recall_curve(y_test, y_pred_proba)

plt.plot(rec, prec)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve')
plt.show()

from sklearn.metrics import roc_auc_score, roc_curve, auc

y_pred_proba = m2.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(y_test, y_pred_proba)
print("AUC:", auc_score)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label = f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # diagonal line = random guessing
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [200, 400],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator = m2,
    param_grid = param_grid,
    scoring = 'roc_auc',   # use AUC because classes are imbalanced; or use recall, f1, accuracy etc.
    cv = 3,                # 3-fold cross-validation
    verbose = 2,
    n_jobs = -1            # use all CPU cores
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation AUC:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\nTest set AUC:", roc_auc_score(y_test, y_pred_proba))
print("Classification report:\n", classification_report(y_test, y_pred))

# Create monthly utilization ratios
df = pd.read_csv("default of credit card clients.csv")

for i in range(1, 7):
    df[f'utilization_{i}'] = df[f'BILL_AMT{i}'] / df['LIMIT_BAL']

# Optionally, create an overall average utilization
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

m3 = xgb.XGBClassifier(
    n_estimators = 500,
    learning_rate = 0.01,
    max_depth = 5,
    subsample = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = pos_weight,
    eval_metric = 'logloss',
    random_state = 1
)

m3.fit(X_train, y_train)
y_pred = m3.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

xgb.plot_importance(m3, max_num_features = 10)
plt.show()

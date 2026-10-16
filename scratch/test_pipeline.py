import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv('data/data.csv', sep=';')
df.columns = df.columns.str.strip()

# Target
y = (df['Target'] == 'Dropout').astype(int)

# Leakage columns to drop
leakage_cols = [c for c in df.columns if '2nd sem' in c] + ['Target']
print("Leakage columns dropped:", leakage_cols)

X = df.drop(columns=leakage_cols)
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Categorical vs Numerical
# Columns that are nominal/categorical codes:
categorical_cols = [
    'Marital status',
    'Application mode',
    'Course',
    'Daytime/evening attendance',
    'Previous qualification',
    'Nacionality',
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    'Displaced',
    'Educational special needs',
    'Debtor',
    'Tuition fees up to date',
    'Gender',
    'Scholarship holder',
    'International'
]

numerical_cols = [c for c in X.columns if c not in categorical_cols]
print(f"Categorical ({len(categorical_cols)}):", categorical_cols)
print(f"Numerical ({len(numerical_cols)}):", numerical_cols)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print("Class counts Train:", y_train.value_counts().to_dict())
print("Class counts Test:", y_test.value_counts().to_dict())

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ],
    remainder='drop'
)

# 1. Logistic Regression
lr_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])
lr_pipe.fit(X_train, y_train)
lr_pred = lr_pipe.predict(X_test)
lr_prob = lr_pipe.predict_proba(X_test)[:, 1]
print("\n--- Logistic Regression ---")
print("Accuracy :", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall   :", recall_score(y_test, lr_pred))
print("F1       :", f1_score(y_test, lr_pred))
print("ROC-AUC  :", roc_auc_score(y_test, lr_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))

# 2. Random Forest
rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=None, class_weight='balanced', random_state=42, n_jobs=-1))
])
rf_pipe.fit(X_train, y_train)
rf_pred = rf_pipe.predict(X_test)
rf_prob = rf_pipe.predict_proba(X_test)[:, 1]
print("\n--- Random Forest ---")
print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall   :", recall_score(y_test, rf_pred))
print("F1       :", f1_score(y_test, rf_pred))
print("ROC-AUC  :", roc_auc_score(y_test, rf_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

# 3. MLP
mlp_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate_init=0.001,
        batch_size=32,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42
    ))
])
sample_weights = compute_sample_weight('balanced', y_train)
mlp_pipe.fit(X_train, y_train, classifier__sample_weight=sample_weights)
mlp_pred = mlp_pipe.predict(X_test)
mlp_prob = mlp_pipe.predict_proba(X_test)[:, 1]
mlp_clf = mlp_pipe.named_steps['classifier']
print("\n--- MLP Classifier ---")
print("Iterations:", mlp_clf.n_iter_)
print("Best loss:", getattr(mlp_clf, 'best_loss_', None))
print("Accuracy :", accuracy_score(y_test, mlp_pred))
print("Precision:", precision_score(y_test, mlp_pred))
print("Recall   :", recall_score(y_test, mlp_pred))
print("F1       :", f1_score(y_test, mlp_pred))
print("ROC-AUC  :", roc_auc_score(y_test, mlp_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, mlp_pred))

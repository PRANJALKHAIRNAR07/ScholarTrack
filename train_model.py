"""
ScholarTrack — Student Academic Risk Prediction
Model Training & Evaluation Pipeline
Configured for data/data.csv (UCI Student Dropout & Academic Success Dataset)
"""

import os
import sys
import json
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import joblib

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

# -----------------------------------------------------------------------------
# Feature Categorization for data/data.csv
# -----------------------------------------------------------------------------
# Excluded leakage columns: all 2nd semester results (future semester grades)
LEAKAGE_COLUMNS = [
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)'
]

CATEGORICAL_FEATURES = [
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

NUMERICAL_FEATURES = [
    'Application order',
    'Previous qualification (grade)',
    'Admission grade',
    'Age at enrollment',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Unemployment rate',
    'Inflation rate',
    'GDP'
]

# Backwards compatibility aliases
DEFAULT_NUMERICAL_COLS = NUMERICAL_FEATURES
DEFAULT_CATEGORICAL_COLS = CATEGORICAL_FEATURES


def load_and_clean_data(data_path: str) -> pd.DataFrame:
    """
    Reads the dataset from disk.
    Tries semicolon (;) first (as in data/data.csv), falling back to comma (,).
    Cleans whitespace and trailing tabs from column names.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    try:
        # Try semicolon first as requested
        df = pd.read_csv(data_path, sep=';')
        if df.shape[1] == 1:
            df = pd.read_csv(data_path, sep=',')
    except Exception:
        df = pd.read_csv(data_path, sep=',')

    # Strip whitespace/tabs from column names (e.g. 'Daytime/evening attendance\t')
    df.columns = df.columns.str.strip()
    return df


def inspect_dataset(data_path: str) -> dict:
    """Inspects the CSV file and returns structural diagnostics."""
    if not os.path.exists(data_path):
        return {
            'exists': False,
            'filepath': data_path,
            'error': f"File not found at {data_path}"
        }

    try:
        df = load_and_clean_data(data_path)
    except Exception as e:
        return {
            'exists': True,
            'filepath': data_path,
            'error': f"Error reading CSV: {str(e)}"
        }

    total_rows, total_cols = df.shape
    columns = list(df.columns)
    non_empty_rows = df.dropna(how='all').shape[0]
    missing_per_col = df.isna().sum().to_dict()
    missing_pct = {col: round((val / total_rows * 100), 2) if total_rows > 0 else 100.0 for col, val in missing_per_col.items()}
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    has_target = 'Target' in columns or 'Exam_Score' in columns
    target_col = 'Target' if 'Target' in columns else ('Exam_Score' if 'Exam_Score' in columns else None)
    target_non_null = int(df[target_col].notnull().sum()) if target_col else 0

    return {
        'exists': True,
        'filepath': data_path,
        'total_rows': total_rows,
        'total_cols': total_cols,
        'non_empty_rows': non_empty_rows,
        'columns': columns,
        'dtypes': dtypes,
        'missing_per_col': missing_per_col,
        'missing_pct': missing_pct,
        'has_target': has_target,
        'target_column': target_col,
        'target_non_null': target_non_null,
        'error': None
    }


def build_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Constructs scikit-learn ColumnTransformer for numerical and categorical features.
    Imputes missing values and standardizes / one-hot encodes.
    """
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ],
        remainder='drop'
    )
    return preprocessor


def train_models(df: pd.DataFrame, output_dir: str = 'models') -> dict:
    """
    Trains Logistic Regression, Random Forest, and MLP Neural Network on data.csv.
    - Delimiter: ';'
    - Target: 'Target' -> At Risk (Dropout = 1), Not at Risk (Graduate / Enrolled = 0)
    - 80:20 stratified train-test split (random_state=42)
    - Preprocessing fit strictly on training data
    - Saves pipelines and model_metadata.json
    """
    if df is None or df.empty:
        raise ValueError("Cannot train model: DataFrame is empty.")

    clean_df = df.copy()
    clean_df.columns = clean_df.columns.str.strip()

    # Determine target definition
    if 'Target' in clean_df.columns:
        # Binary target: Dropout = 1 (At Risk), Graduate/Enrolled = 0 (Not at Risk)
        clean_df = clean_df.dropna(subset=['Target'])
        y = (clean_df['Target'] == 'Dropout').astype(int)
        target_name = 'Target'
    elif 'Exam_Score' in clean_df.columns:
        # Fallback for earlier StudentPerformanceFactors dataset
        clean_df = clean_df.dropna(subset=['Exam_Score'])
        y = (clean_df['Exam_Score'] < 60.0).astype(int)
        target_name = 'Exam_Score'
    else:
        raise ValueError("Neither 'Target' nor 'Exam_Score' found in dataset.")

    # 1. Print class distribution before splitting
    total_samples = len(y)
    class_counts_before = y.value_counts().to_dict()
    at_risk_before = int(class_counts_before.get(1, 0))
    not_at_risk_before = int(class_counts_before.get(0, 0))
    pct_at_risk_before = (at_risk_before / total_samples) * 100
    pct_not_at_risk_before = (not_at_risk_before / total_samples) * 100

    print("\n" + "=" * 70)
    print("  CLASS DISTRIBUTION BEFORE SPLITTING")
    print("=" * 70)
    print(f"  Total records: {total_samples:,}")
    print(f"  • Class 0 [Not at Risk (Graduate / Enrolled)]: {not_at_risk_before:,} ({pct_not_at_risk_before:.2f}%)")
    print(f"  • Class 1 [At Risk (Dropout)]               : {at_risk_before:,} ({pct_at_risk_before:.2f}%)")
    print("-" * 70)

    # 2. Exclude data leakage features
    # Exclude future semester grades and results (all 2nd semester curricular units)
    leakage_in_df = [c for c in clean_df.columns if any(leak.lower() in c.lower() for leak in ['2nd sem', 'second sem'])]
    cols_to_drop = list(set(leakage_in_df + [target_name]))
    if 'Academic_Risk' in clean_df.columns:
        cols_to_drop.append('Academic_Risk')

    X = clean_df.drop(columns=[c for c in cols_to_drop if c in clean_df.columns])

    print(f"  Data Leakage Prevention:")
    print(f"  • Excluded {len(leakage_in_df)} future semester features: {leakage_in_df}")
    print(f"  • Remaining input features: {X.shape[1]}")

    # Determine numerical and categorical columns present in X
    num_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    # In case there are remaining columns not in predefined lists, allocate them by dtype
    unassigned = [c for c in X.columns if c not in num_cols and c not in cat_cols]
    for col in unassigned:
        if clean_df[col].dtype in ['float64', 'int64'] and clean_df[col].nunique() > 10:
            num_cols.append(col)
        else:
            cat_cols.append(col)

    print(f"  • Numerical features ({len(num_cols)}): {num_cols}")
    print(f"  • Categorical features ({len(cat_cols)}): {cat_cols}")

    # 3. 80:20 Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    train_total = len(y_train)
    test_total = len(y_test)
    train_counts = y_train.value_counts().to_dict()
    test_counts = y_test.value_counts().to_dict()

    print("\n" + "=" * 70)
    print("  CLASS DISTRIBUTION AFTER 80:20 STRATIFIED SPLIT")
    print("=" * 70)
    print(f"  Training Set ({train_total:,} records):")
    print(f"  • Class 0 [Not at Risk]: {train_counts.get(0, 0):,} ({train_counts.get(0, 0)/train_total*100:.2f}%)")
    print(f"  • Class 1 [At Risk]    : {train_counts.get(1, 0):,} ({train_counts.get(1, 0)/train_total*100:.2f}%)")
    print(f"  Test Set ({test_total:,} records):")
    print(f"  • Class 0 [Not at Risk]: {test_counts.get(0, 0):,} ({test_counts.get(0, 0)/test_total*100:.2f}%)")
    print(f"  • Class 1 [At Risk]    : {test_counts.get(1, 0):,} ({test_counts.get(1, 0)/test_total*100:.2f}%)")
    print("-" * 70)

    # 4. Build Preprocessor (Fit strictly on training data inside Pipeline)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # -------------------------------------------------------------------------
    # Model 1: Logistic Regression
    # -------------------------------------------------------------------------
    print("\n[1/3] Training Logistic Regression...")
    lr_clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        solver='lbfgs'
    )
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', lr_clf)
    ])
    lr_pipeline.fit(X_train, y_train)

    lr_y_pred = lr_pipeline.predict(X_test)
    lr_y_prob = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_cm = confusion_matrix(y_test, lr_y_pred)
    lr_metrics = {
        'model_name': 'Logistic Regression',
        'accuracy': float(accuracy_score(y_test, lr_y_pred)),
        'precision': float(precision_score(y_test, lr_y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, lr_y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, lr_y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, lr_y_prob)),
        'confusion_matrix': lr_cm.tolist(),
        'classification_report': classification_report(y_test, lr_y_pred, output_dict=True, zero_division=0),
        'hyperparameters': {
            'max_iter': 1000,
            'class_weight': 'balanced',
            'random_state': 42,
            'solver': 'lbfgs'
        }
    }

    # Extract Logistic Regression coefficients / odds ratios
    feature_importances = []
    try:
        prep_fitted = lr_pipeline.named_steps['preprocessor']
        ohe_cols = list(prep_fitted.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols))
        all_feature_names = num_cols + ohe_cols
        coefs = lr_pipeline.named_steps['classifier'].coef_[0]
        if len(all_feature_names) == len(coefs):
            for feat, coef in zip(all_feature_names, coefs):
                feature_importances.append({
                    'feature': feat,
                    'coefficient': float(coef),
                    'odds_ratio': float(np.exp(coef))
                })
            feature_importances.sort(key=lambda x: abs(x['coefficient']), reverse=True)
    except Exception as e:
        print(f"Notice: Could not extract coefficients: {e}")

    # -------------------------------------------------------------------------
    # Model 2: Random Forest
    # -------------------------------------------------------------------------
    print("[2/3] Training Random Forest...")
    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf_clf)
    ])
    rf_pipeline.fit(X_train, y_train)

    rf_y_pred = rf_pipeline.predict(X_test)
    rf_y_prob = rf_pipeline.predict_proba(X_test)[:, 1]
    rf_cm = confusion_matrix(y_test, rf_y_pred)
    rf_metrics = {
        'model_name': 'Random Forest',
        'accuracy': float(accuracy_score(y_test, rf_y_pred)),
        'precision': float(precision_score(y_test, rf_y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, rf_y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, rf_y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, rf_y_prob)),
        'confusion_matrix': rf_cm.tolist(),
        'classification_report': classification_report(y_test, rf_y_pred, output_dict=True, zero_division=0),
        'hyperparameters': {
            'n_estimators': 100,
            'max_depth': None,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
    }

    # -------------------------------------------------------------------------
    # Model 3: MLP Neural Network (scikit-learn MLPClassifier)
    # -------------------------------------------------------------------------
    print("[3/3] Training MLP Neural Network...")
    mlp_clf = MLPClassifier(
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
    )
    mlp_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', mlp_clf)
    ])
    # Handle class imbalance via sample weights on training set
    sample_weights = compute_sample_weight('balanced', y_train)
    mlp_pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)

    mlp_fitted_clf = mlp_pipeline.named_steps['classifier']
    mlp_iterations = int(mlp_fitted_clf.n_iter_)
    mlp_converged = bool(mlp_iterations < 200 or getattr(mlp_fitted_clf, 'loss_', 1.0) < 1e-4)

    mlp_y_pred = mlp_pipeline.predict(X_test)
    mlp_y_prob = mlp_pipeline.predict_proba(X_test)[:, 1]
    mlp_cm = confusion_matrix(y_test, mlp_y_pred)
    mlp_metrics = {
        'model_name': 'Multi-Layer Perceptron (MLP)',
        'accuracy': float(accuracy_score(y_test, mlp_y_pred)),
        'precision': float(precision_score(y_test, mlp_y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, mlp_y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, mlp_y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, mlp_y_prob)),
        'confusion_matrix': ml_cm if (ml_cm := mlp_cm.tolist()) else [],
        'classification_report': classification_report(y_test, mlp_y_pred, output_dict=True, zero_division=0),
        'hyperparameters': {
            'hidden_layer_sizes': [64, 32],
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'learning_rate_init': 0.001,
            'batch_size': 32,
            'max_iter': 200,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 15,
            'random_state': 42,
            'sample_weighting': 'balanced (inverse frequency)'
        },
        'convergence': {
            'converged': mlp_converged,
            'actual_iterations': mlp_iterations,
            'stopped_by': 'early_stopping' if mlp_iterations < 200 else 'max_iter'
        }
    }

    # 5. Extract default median/mode profile from training set for single-student inference
    defaults = {}
    for col in num_cols:
        defaults[col] = float(X_train[col].median())
    for col in cat_cols:
        defaults[col] = int(X_train[col].mode()[0]) if pd.api.types.is_integer_dtype(X_train[col]) else str(X_train[col].mode()[0])

    # 6. Save artifacts
    os.makedirs(output_dir, exist_ok=True)
    lr_path = os.path.join(output_dir, 'academic_risk_pipeline.joblib')
    rf_path = os.path.join(output_dir, 'academic_risk_rf_pipeline.joblib')
    mlp_path = os.path.join(output_dir, 'academic_risk_mlp_pipeline.joblib')
    meta_path = os.path.join(output_dir, 'model_metadata.json')

    joblib.dump(lr_pipeline, lr_path)
    joblib.dump(rf_pipeline, rf_path)
    joblib.dump(mlp_pipeline, mlp_path)

    metadata = {
        'trained_at': datetime.now().isoformat(),
        'dataset_file': 'data/data.csv',
        'delimiter': ';',
        'target_column': 'Target',
        'target_definition': 'At Risk = Dropout (1), Not at Risk = Graduate or Enrolled (0)',
        'total_samples': total_samples,
        'train_samples': train_total,
        'test_samples': test_total,
        'features_total_count': X.shape[1],
        'leakage_features_excluded': leakage_in_df,
        'class_distribution_before_split': {
            'total': total_samples,
            'at_risk_count': at_risk_before,
            'not_at_risk_count': not_at_risk_before,
            'at_risk_pct': round(pct_at_risk_before, 2),
            'not_at_risk_pct': round(pct_not_at_risk_before, 2)
        },
        'class_distribution_train': {
            'total': train_total,
            'at_risk_count': int(train_counts.get(1, 0)),
            'not_at_risk_count': int(train_counts.get(0, 0)),
            'at_risk_pct': round(train_counts.get(1, 0) / train_total * 100, 2),
            'not_at_risk_pct': round(train_counts.get(0, 0) / train_total * 100, 2)
        },
        'class_distribution_test': {
            'total': test_total,
            'at_risk_count': int(test_counts.get(1, 0)),
            'not_at_risk_count': int(test_counts.get(0, 0)),
            'at_risk_pct': round(test_counts.get(1, 0) / test_total * 100, 2),
            'not_at_risk_pct': round(test_counts.get(0, 0) / test_total * 100, 2)
        },
        'features': {
            'numerical': num_cols,
            'categorical': cat_cols
        },
        'feature_defaults': defaults,
        'logistic_regression': lr_metrics,
        'random_forest': rf_metrics,
        'mlp': mlp_metrics,
        'feature_analysis': feature_importances,
        'sequential_lstm_audit': {
            'dataset_nature': 'Static cross-sectional (4,424 single-row student records)',
            'has_temporal_dimension': False,
            'lstm_status': 'Not trained (no genuine longitudinal/sequential observations in data.csv)',
            'policy': 'Per scientific standard and project requirements, static attributes are not converted into artificial time-steps.'
        }
    }

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="ScholarTrack Model Training Pipeline")
    parser.add_argument('--data-path', type=str, default='data/data.csv',
                        help='Path to student CSV dataset (default: data/data.csv)')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to store trained model artifacts')
    args = parser.parse_args()

    print("=" * 70)
    print("  ScholarTrack — Student Academic Risk Prediction: Model Pipeline")
    print("=" * 70)
    print(f"Target dataset: {args.data_path}")

    # Inspect dataset
    report = inspect_dataset(args.data_path)
    if not report['exists']:
        print(f"[ERROR] {report['error']}")
        sys.exit(1)

    print("\n[DATASET DIAGNOSTICS]")
    print(f"  • File path: {report['filepath']}")
    print(f"  • Total rows: {report['total_rows']:,}")
    print(f"  • Total columns: {report['total_cols']:,}")
    print(f"  • Target column: {report['target_column']}")

    try:
        df = load_and_clean_data(args.data_path)
        metadata = train_models(df, output_dir=args.output_dir)

        print("\n" + "=" * 70)
        print("  TRAINING & EVALUATION COMPLETED ON UNTOUCHED TEST SET (885 SAMPLES)")
        print("=" * 70)

        lr_m = metadata['logistic_regression']
        print(f"\n[1] Logistic Regression (Primary):")
        print(f"    - Accuracy : {lr_m['accuracy']*100:.2f}%")
        print(f"    - Precision: {lr_m['precision']:.4f}")
        print(f"    - Recall   : {lr_m['recall']:.4f} (Sensitivity on At Risk)")
        print(f"    - F1-Score : {lr_m['f1_score']:.4f}")
        print(f"    - ROC-AUC  : {lr_m['roc_auc']:.4f}")
        print(f"    - Confusion Matrix: {lr_m['confusion_matrix']}")

        rf_m = metadata['random_forest']
        print(f"\n[2] Random Forest (Comparison):")
        print(f"    - Accuracy : {rf_m['accuracy']*100:.2f}%")
        print(f"    - Precision: {rf_m['precision']:.4f}")
        print(f"    - Recall   : {rf_m['recall']:.4f} (Sensitivity on At Risk)")
        print(f"    - F1-Score : {rf_m['f1_score']:.4f}")
        print(f"    - ROC-AUC  : {rf_m['roc_auc']:.4f}")
        print(f"    - Confusion Matrix: {rf_m['confusion_matrix']}")

        mlp_m = metadata['mlp']
        print(f"\n[3] Multi-Layer Perceptron - MLP:")
        print(f"    - Converged: {mlp_m['convergence']['converged']} ({mlp_m['convergence']['actual_iterations']} iterations via {mlp_m['convergence']['stopped_by']})")
        print(f"    - Accuracy : {mlp_m['accuracy']*100:.2f}%")
        print(f"    - Precision: {mlp_m['precision']:.4f}")
        print(f"    - Recall   : {mlp_m['recall']:.4f} (Sensitivity on At Risk)")
        print(f"    - F1-Score : {mlp_m['f1_score']:.4f}")
        print(f"    - ROC-AUC  : {mlp_m['roc_auc']:.4f}")
        print(f"    - Confusion Matrix: {mlp_m['confusion_matrix']}")

        print(f"\nModel artifacts successfully serialized to: {args.output_dir}/")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR DURING TRAINING]: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

# 🎓 ScholarTrack — Student Academic Risk Prediction

An early warning machine learning platform designed to identify higher-education students at risk of dropout **prior to future semester outcomes**, enabling timely, targeted, and empathetic institutional interventions.

---

## 📌 Problem Formulation & Early Warning Objective

Student attrition in higher education leads to unfulfilled academic potential, significant personal debt, and severe institutional loss. Traditional academic reviews take place after final semester exams or upon official withdrawal—when intervention is no longer possible.

**ScholarTrack** addresses this by formulating early dropout detection as a supervised binary classification task:
- **Target Variable (`Target`)**:
  - `Dropout` $\rightarrow$ **At Risk (Class 1)**
  - `Graduate` or `Enrolled` $\rightarrow$ **Not at Risk (Class 0)**
- **Early-Risk Boundary**: Models are trained strictly on pre-enrollment characteristics (demographics, entrance grades, socio-economic factors) and **1st-semester academic results**, deliberately excluding 2nd-semester and subsequent results to guarantee early intervention capability.

---

## 🔍 Dataset Audit & Leakage Prevention

The system is configured and evaluated on the uploaded dataset: [`data/data.csv`](data/data.csv) (semicolon `;` delimiter).

| Property | Detail |
| :--- | :--- |
| **Dataset File** | `data/data.csv` (Delimiter: `;`) |
| **Total Records** | **4,424 student records** across 37 raw columns |
| **Target Distribution** | **Dropout (At Risk): 1,421 (32.12%)** \| **Graduate / Enrolled (Not at Risk): 3,003 (67.88%)** |
| **Train / Test Split** | **80:20 Stratified Split** (`random_state=42`) |
| **Train Cohort** | **3,539 records** (Class 0: 2,402 [67.87%], Class 1: 1,137 [32.13%]) |
| **Held-Out Test Cohort** | **885 records** (Class 0: 601 [67.91%], Class 1: 284 [32.09%]) |
| **Features Used** | **30 features** (13 numerical, 17 categorical) |

### Exact Features Excluded to Prevent Data Leakage
To ensure realistic early warning before student failure occurs, all 6 future (2nd-semester) outcome features were strictly removed:
1. `Curricular units 2nd sem (credited)`
2. `Curricular units 2nd sem (enrolled)`
3. `Curricular units 2nd sem (evaluations)`
4. `Curricular units 2nd sem (approved)`
5. `Curricular units 2nd sem (grade)`
6. `Curricular units 2nd sem (without evaluations)`
*(Plus the target column `Target`)*

---

## 🏗️ Preprocessing & Model Architecture

All transformations are encapsulated inside a scikit-learn `Pipeline` and `ColumnTransformer` to prevent data leakage. Preprocessing transformations are **fit strictly on the training partition**:

```
Input Features (30 Pre-Leakage Features)
   │
   ├── Numerical Features (13 features: 1st sem units, grades, admission score, age, macroeconomic rates)
   │     └── SimpleImputer(strategy='median') ──► StandardScaler()
   │
   ├── Categorical Features (17 features: tuition up to date, debtor, scholarship, gender, course, parents' jobs)
   │     └── SimpleImputer(strategy='most_frequent') ──► OneHotEncoder(handle_unknown='ignore', sparse_output=False)
   │
   └── ColumnTransformer
         ├── [1] Logistic Regression (Primary Model, balanced class weights)
         ├── [2] Random Forest Classifier (Comparison Model, 100 estimators, balanced)
         └── [3] Multi-Layer Perceptron - MLP (Deep Learning, 64x32 ReLU, balanced sample weights)
```

---

## ⚙️ Model Hyperparameters

### 1. Logistic Regression (Primary Model)
- `max_iter`: `1000`
- `class_weight`: `'balanced'`
- `solver`: `'lbfgs'` (default solver, converged without warnings)
- `random_state`: `42`

### 2. Random Forest (Comparison Model)
- `n_estimators`: `100`
- `max_depth`: `None`
- `class_weight`: `'balanced'`
- `n_jobs`: `-1`
- `random_state`: `42`

### 3. Multi-Layer Perceptron (MLP Neural Network)
- `hidden_layer_sizes`: `(64, 32)`
- `activation`: `'relu'`
- `solver`: `'adam'`
- `alpha`: `0.001` (L2 regularization penalty)
- `learning_rate_init`: `0.001`
- `batch_size`: `32`
- `max_iter`: `200` (maximum epochs)
- `early_stopping`: `True`
- `validation_fraction`: `0.1`
- `n_iter_no_change`: `15`
- `random_state`: `42`
- `sample_weight`: Balanced inverse-frequency sample weighting applied to training split (`compute_sample_weight('balanced', y_train)`).
- **Convergence Status**: Converged via early stopping at **iteration 27** (validation loss plateaued after epoch 12, ran for 15 patience epochs, well below max iterations of 200).

---

## 📊 Test-Set Evaluation Results (Held-Out 885 Samples)

All three models were evaluated on the exact same untouched test set (601 Not at Risk, 284 At Risk):

| Metric | Logistic Regression (Primary) | Multi-Layer Perceptron (MLP) | Random Forest (Comparison) |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **85.88%** | **81.24%** | **85.54%** |
| **Recall / Sensitivity (At Risk)** | **80.28%** *(228 / 284 caught)* | **82.75%** *(235 / 284 caught)* | **75.00%** *(213 / 284 caught)* |
| **Precision (At Risk)** | **76.77%** | **66.76%** | **78.89%** |
| **F1-Score (At Risk)** | **0.7849** | **0.7390** | **0.7690** |
| **ROC-AUC** | **0.9126** | **0.8983** | **0.9076** |

### Confusion Matrix Breakdown (885 Test Records)
- **Logistic Regression:** $\text{TN} = 532$, $\text{FP} = 69$, $\text{FN} = 56$, $\text{TP} = 228$
- **MLP Neural Network:** $\text{TN} = 484$, $\text{FP} = 117$, $\text{FN} = 49$, $\text{TP} = 235$ *(Lowest false negatives / highest recall)*
- **Random Forest:** $\text{TN} = 544$, $\text{FP} = 57$, $\text{FN} = 71$, $\text{TP} = 213$

> [!TIP]
> **Recall Priority in Student Retention:**
> In early-warning educational systems, false negatives ($\text{FN}$) represent vulnerable students who drop out without receiving assistance. The **MLP** achieves the highest sensitivity, catching **82.75%** of at-risk students, while **Logistic Regression** delivers the best balanced performance (**80.28% recall, 85.88% accuracy, 0.9126 ROC-AUC**).

---

## ⏳ Sequential / Longitudinal Modeling & LSTM Audit

In strict compliance with project guidelines: **we do not claim LSTM is trained or implemented on `data/data.csv`**.

### 1. Dataset Nature & Scientific Boundary
- `data/data.csv` is a **static cross-sectional dataset** (each of the 4,424 rows represents a single student snapshot with cumulative 1st-semester aggregate statistics).
- There are no weekly timestamps, no sequential LMS logs, and no repeated measurements over time.
- Static columns (e.g., age, admission grade, approved units) **cannot be treated as sequential time steps**. Doing so induces artificial order and produces meaningless recurrent parameters.

### 2. Recommended Longitudinal Datasets for LSTM
When institutional sequential data is available, suitable public benchmarks include:
1. **OULAD (Open University Learning Analytics Dataset):** 32,593 students with daily clickstream activity and longitudinal assignment submissions across 39 weeks.
2. **Cortez & Silva Portuguese Student Dataset:** 3 distinct evaluation periods ($G_1, G_2, G_3$) throughout the academic year.

### 3. LSTM Architectural Specification
When 3D sequential tensors $(N, T, D)$ are available:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dropout, Dense

model = Sequential([
    Masking(mask_value=0.0, input_shape=(T, D)),
    LSTM(64, return_sequences=True, dropout=0.2),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
```

---

## 🚀 Quickstart Guide (Windows PowerShell)

### 1. Clone / Navigate to Directory
```powershell
cd c:\Users\pranj\OneDrive\Desktop\ScholarTrack
```

### 2. Train and Serialize Models
```powershell
python train_model.py
```
This script reads `data/data.csv` (using delimiter `;`), splits the data with stratification, trains all 3 models, outputs diagnostics and evaluation metrics, and serializes artifacts into `models/`.

### 3. Launch the Streamlit App
```powershell
streamlit run app.py
```
Access the application at `http://localhost:8501`.

---

## 🖥️ Streamlit Dashboard Walkthrough

1. **📊 Dashboard:**
   - Real-time KPI cards for student population (4,424 records), at-risk count (1,421 / 32.1%), on-track count (3,003 / 67.9%), and test recall.
   - Interactive Plotly visualizations for cohort target distribution and 1st-semester performance correlation.
2. **🎯 Predict Risk:**
   - Interactive prediction form capturing 1st-semester approved units, grades, evaluations, admission scores, tuition status, debt, and demographics.
   - **Model Selector:** Switch dynamically between Logistic Regression, MLP Neural Net, and Random Forest.
   - Real-time prediction with probability bar and risk alert pill.
   - **Personalized Academic Support Plan:** Formatted in high-contrast dark charcoal text (`#111827`) with prioritized actionable interventions.
3. **📈 Dataset Insights & Upload:**
   - Deep inspection of `data/data.csv` (and any other dataset in `data/`), schema diagnostics, and missing value distribution.
   - Dynamic CSV uploader supporting both `;` and `,` delimiters with instant retraining.
4. **🧠 Model Performance:**
   - Head-to-head comparison table across all 3 models using genuine held-out test set metrics.
   - 3 interactive Plotly confusion matrix heatmaps.
   - Logistic regression feature importance / odds ratios chart.
   - Hyperparameter & convergence audit.
   - Longitudinal modeling & LSTM analysis.

---

## ⚠️ Limitations

1. **Early Semester Window:** Risk assessment depends heavily on 1st-semester course approvals; accuracy is highest after initial 1st-semester grades are recorded.
2. **Institutional Specificity:** The dataset originates from a Portuguese polytechnic institution; while macro features generalize well, grading scales (0–20) and course codes may require calibration for other university systems.
3. **Static Representation:** Dataset lacks sub-semester temporal granularity (e.g., weekly LMS logins or attendance logs).

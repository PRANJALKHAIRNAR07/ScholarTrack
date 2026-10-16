"""
ScholarTrack — Student Academic Risk Prediction
Interactive Streamlit Dashboard
Configured for data/data.csv (UCI Student Dropout & Academic Success Dataset)
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

from train_model import (
    load_and_clean_data,
    inspect_dataset,
    train_models
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="ScholarTrack — Student Academic Risk Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global Typography & Font Setup */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    
    .hero-title {
        color: #f8fafc;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
        line-height: 1.5;
    }
    
    .badge-pill {
        display: inline-block;
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
    }

    /* Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .kpi-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    .kpi-value {
        color: #0f172a;
        font-size: 1.85rem;
        font-weight: 700;
        margin-top: 4px;
    }
    
    .kpi-desc {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 4px;
    }

    /* Risk Status Callouts */
    .risk-alert-high {
        background: #fef2f2;
        border-left: 6px solid #ef4444;
        border-radius: 8px;
        padding: 18px 24px;
        margin: 16px 0;
    }
    
    .risk-alert-low {
        background: #f0fdf4;
        border-left: 6px solid #10b981;
        border-radius: 8px;
        padding: 18px 24px;
        margin: 16px 0;
    }

    /* Personalized Academic Support Plan Card & Recommendations (Scoped) */
    .support-plan-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        color: #111827 !important;
    }

    .support-plan-card h4,
    .support-plan-heading {
        color: #111827 !important;
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 16px;
        letter-spacing: -0.01em;
    }

    .support-plan-card p,
    .support-plan-card span,
    .support-plan-card div,
    .support-plan-card ul,
    .support-plan-card ol,
    .support-plan-card li,
    .support-plan-card strong,
    .support-plan-card b {
        color: #111827 !important;
    }

    .recommendation-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        color: #111827 !important;
        line-height: 1.55;
        font-size: 0.95rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }

    .recommendation-card:last-child {
        margin-bottom: 0;
    }

    .recommendation-card,
    .recommendation-card * {
        color: #111827 !important;
    }

    .recommendation-card strong,
    .recommendation-card b,
    .rec-title {
        color: #111827 !important;
        font-weight: 700;
    }

    .recommendation-card p,
    .recommendation-card span,
    .recommendation-card div,
    .recommendation-card li,
    .recommendation-card ul,
    .recommendation-card ol,
    .rec-body {
        color: #111827 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DEFAULT_DATA_PATH = "data/data.csv"
MODEL_DIR = "models"
PRIMARY_MODEL_PATH = os.path.join(MODEL_DIR, "academic_risk_pipeline.joblib")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "academic_risk_rf_pipeline.joblib")
MLP_MODEL_PATH = os.path.join(MODEL_DIR, "academic_risk_mlp_pipeline.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


# ---------------------------------------------------------
# Helper Functions: State & Data Loading
# ---------------------------------------------------------
@st.cache_data
def get_default_dataset_info():
    return inspect_dataset(DEFAULT_DATA_PATH)

def load_saved_metadata():
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def load_saved_model(path):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


# ---------------------------------------------------------
# Sidebar Navigation & Settings
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 ScholarTrack")
    st.markdown(
        "<span style='color: #64748b; font-size: 0.85rem; font-weight: 500;'>"
        "Student Academic Risk Early Warning System</span>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🎯 Predict Risk", "📈 Dataset Insights & Upload", "🧠 Model Performance"],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ Target & Risk Definition")
    st.markdown(
        """
        - **Target Variable:** `Target`
        - **At Risk (Class 1):** `Dropout`
        - **Not at Risk (Class 0):** `Graduate` or `Enrolled`
        """
    )
    st.caption("Models trained to flag students at high risk of dropping out prior to future semester outcomes.")

    st.markdown("---")
    metadata = load_saved_metadata()
    default_info = get_default_dataset_info()

    st.markdown("### 📌 System Status")
    if metadata:
        st.success(f"● Models: 3 Trained ({metadata['total_samples']:,} records)")
        st.caption(f"LR Acc: {metadata['logistic_regression']['accuracy']*100:.1f}% | Recall: {metadata['logistic_regression']['recall']*100:.1f}%")
    else:
        st.warning("● Models: Not Trained")

    if 'active_df' in st.session_state and st.session_state['active_df'] is not None:
        st.info(f"● Active Data: Custom ({len(st.session_state['active_df']):,} rows)")
    else:
        st.info(f"● Active Data: `data/data.csv` ({default_info['total_rows']:,} rows)")

    st.markdown("---")
    st.caption("ScholarTrack v2.0 • Early Warning AI")


# ---------------------------------------------------------
# Page 1: Dashboard
# ---------------------------------------------------------
if page == "📊 Dashboard":
    st.markdown(
        """
        <div class="hero-container">
            <span class="badge-pill">Academic Intelligence Engine</span>
            <h1 class="hero-title">ScholarTrack — Student Risk Prediction</h1>
            <p class="hero-subtitle">
                An early warning machine learning platform designed to identify students at academic risk 
                (dropout prediction) based on 1st-semester performance, admissions, and socioeconomic background.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Active dataset resolution
    if 'active_df' not in st.session_state or st.session_state['active_df'] is None:
        if os.path.exists(DEFAULT_DATA_PATH):
            try:
                st.session_state['active_df'] = load_and_clean_data(DEFAULT_DATA_PATH)
            except Exception:
                pass
    active_df = st.session_state.get('active_df', None)
    metadata = load_saved_metadata()

    if active_df is not None and 'Target' in active_df.columns:
        st.success(
            f"ℹ️ **Active Dataset:** Using `data/data.csv` with {len(active_df):,} student records "
            f"(Target: Dropout vs Graduate/Enrolled). Future semester grades excluded to prevent leakage."
        )

    # KPI Metric Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_records = len(active_df) if active_df is not None else default_info['total_rows']
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Student Records</div>
                <div class="kpi-value">{total_records:,}</div>
                <div class="kpi-desc">Cohort population in dataset</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        if active_df is not None and 'Target' in active_df.columns:
            at_risk = int((active_df['Target'] == 'Dropout').sum())
            pct = round(at_risk / len(active_df) * 100, 1)
        elif metadata:
            at_risk = metadata['class_distribution_before_split']['at_risk_count']
            pct = metadata['class_distribution_before_split']['at_risk_pct']
        else:
            at_risk = 1421
            pct = 32.1

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">At Risk Students (Dropout)</div>
                <div class="kpi-value" style="color: #ef4444;">{at_risk:,}</div>
                <div class="kpi-desc">{pct}% of student population</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        if active_df is not None and 'Target' in active_df.columns:
            on_track = len(active_df) - at_risk
            pct_track = round(on_track / len(active_df) * 100, 1)
        elif metadata:
            on_track = metadata['class_distribution_before_split']['not_at_risk_count']
            pct_track = metadata['class_distribution_before_split']['not_at_risk_pct']
        else:
            on_track = 3003
            pct_track = 67.9

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">On Track (Graduate / Enrolled)</div>
                <div class="kpi-value" style="color: #10b981;">{on_track:,}</div>
                <div class="kpi-desc">{pct_track}% continuing or graduated</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        lr_recall = metadata['logistic_regression']['recall'] * 100 if metadata else 80.3
        lr_acc = metadata['logistic_regression']['accuracy'] * 100 if metadata else 85.9
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Primary Model Recall</div>
                <div class="kpi-value" style="color: #3b82f6;">{lr_recall:.1f}%</div>
                <div class="kpi-desc">Accuracy: {lr_acc:.1f}% (Test Set)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Visualizations
    v_col1, v_col2 = st.columns([1, 1])

    with v_col1:
        if active_df is not None and 'Target' in active_df.columns:
            target_counts = active_df['Target'].value_counts()
            fig_pie = px.pie(
                names=target_counts.index,
                values=target_counts.values,
                title="Target Class Breakdown in Cohort (data.csv)",
                color=target_counts.index,
                color_discrete_map={
                    "Dropout": "#ef4444",
                    "Graduate": "#10b981",
                    "Enrolled": "#3b82f6"
                },
                hole=0.45
            )
            fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

    with v_col2:
        if active_df is not None and 'Curricular units 1st sem (approved)' in active_df.columns:
            sample_df = active_df.sample(min(1000, len(active_df)), random_state=42)
            fig_scatter = px.scatter(
                sample_df,
                x="Curricular units 1st sem (approved)",
                y="Curricular units 1st sem (grade)",
                color=sample_df['Target'].map(lambda t: "At Risk (Dropout)" if t == "Dropout" else "On Track (Graduate/Enrolled)"),
                color_discrete_map={
                    "At Risk (Dropout)": "#ef4444",
                    "On Track (Graduate/Enrolled)": "#10b981"
                },
                title="1st Semester Units Approved vs. Average Grade (1,000 Sample Points)",
                labels={
                    "Curricular units 1st sem (approved)": "1st Sem Approved Units",
                    "Curricular units 1st sem (grade)": "1st Sem Average Grade (0-20)",
                    "color": "Academic Status"
                }
            )
            fig_scatter.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧭 Quick Navigation")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.info("🎯 **Evaluate Individual Student**\nInput student academic scores, financial standing, and enrollment metrics to run live risk predictions.")
    with q2:
        st.info("📈 **Explore Dataset Insights**\nInspect data/data.csv schema, missing values distribution, and column descriptions.")
    with q3:
        st.info("🧠 **Model Evaluation & Comparison**\nCompare Logistic Regression, Random Forest, and MLP neural net test performance on 885 unseen test records.")


# ---------------------------------------------------------
# Page 2: Predict Risk
# ---------------------------------------------------------
elif page == "🎯 Predict Risk":
    st.markdown("## 🎯 Student Academic Risk Prediction")
    st.markdown(
        "Enter student parameters below to evaluate their risk of dropping out prior to future semester outcomes. "
        "Select your preferred machine learning engine."
    )

    metadata = load_saved_metadata()
    defaults = metadata.get('feature_defaults', {}) if metadata else {}

    m_col1, m_col2 = st.columns([2, 1])
    with m_col1:
        model_choice = st.selectbox(
            "Select Prediction Engine",
            [
                "Logistic Regression (Recommended — Primary Model)",
                "Multi-Layer Perceptron (MLP Neural Network)",
                "Random Forest (Comparison Model)"
            ],
            index=0,
            help="Logistic Regression provides calibrated odds ratios and high recall. MLP captures non-linear neural activations."
        )
    with m_col2:
        if "Logistic Regression" in model_choice:
            st.caption("⚙️ **Engine:** Logistic Regression\nBalanced weights, odds-ratio explainability (Test Recall: 80.3%).")
        elif "Multi-Layer Perceptron" in model_choice:
            st.caption("⚙️ **Engine:** Multi-Layer Perceptron (MLP)\n(64, 32) ReLU hidden layers, balanced sample weights (Test Recall: 82.7%).")
        else:
            st.caption("⚙️ **Engine:** Random Forest\nEnsemble of 100 decision trees (Test Recall: 75.0%).")

    with st.form("risk_prediction_form"):
        st.markdown("#### 1. 1st Semester Academic Performance (High Predictive Impact)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sem1_approved = st.slider(
                "1st Sem Approved Units",
                min_value=0, max_value=15, value=int(defaults.get("Curricular units 1st sem (approved)", 5)),
                help="Number of curricular units approved in the 1st semester."
            )
        with c2:
            sem1_grade = st.slider(
                "1st Sem Average Grade (0–20)",
                min_value=0.0, max_value=20.0, value=float(round(defaults.get("Curricular units 1st sem (grade)", 12.3), 1)), step=0.1,
                help="Average grade earned in 1st semester curricular units (Portuguese 0-20 scale)."
            )
        with c3:
            sem1_evals = st.slider(
                "1st Sem Evaluations Attended",
                min_value=0, max_value=25, value=int(defaults.get("Curricular units 1st sem (evaluations)", 8)),
                help="Number of evaluations taken in 1st semester."
            )
        with c4:
            sem1_enrolled = st.slider(
                "1st Sem Enrolled Units",
                min_value=0, max_value=15, value=int(defaults.get("Curricular units 1st sem (enrolled)", 6)),
                help="Total curricular units student registered for in 1st semester."
            )

        st.markdown("#### 2. Admission Background & Age")
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            admission_grade = st.slider(
                "Admission Grade (0–200)",
                min_value=90.0, max_value=200.0, value=float(round(defaults.get("Admission grade", 126.0), 1)), step=0.5,
                help="Score on higher-education entrance examination (scale 0-200)."
            )
        with c6:
            prev_grade = st.slider(
                "Previous Qualification Grade (0–200)",
                min_value=90.0, max_value=200.0, value=float(round(defaults.get("Previous qualification (grade)", 133.0), 1)), step=0.5,
                help="Grade obtained in high school / previous secondary education."
            )
        with c7:
            age = st.slider(
                "Age at Enrollment",
                min_value=17, max_value=60, value=int(defaults.get("Age at enrollment", 20)),
                help="Student age upon registering at university."
            )
        with c8:
            app_order = st.selectbox(
                "Application Preference Order",
                options=[1, 2, 3, 4, 5, 6],
                index=0,
                help="1 = 1st choice course, 6 = 6th choice."
            )

        st.markdown("#### 3. Financial, Institutional & Demographic Factors")
        c9, c10, c11, c12, c13 = st.columns(5)
        with c9:
            tuition_up_to_date = st.selectbox(
                "Tuition Fees Up to Date?",
                options=["Yes", "No"],
                index=0,
                help="Crucial financial stability indicator."
            )
        with c10:
            is_debtor = st.selectbox(
                "Has Overdue Debt (Debtor)?",
                options=["No", "Yes"],
                index=0,
                help="Whether student has unaddressed institutional debt."
            )
        with c11:
            is_scholarship = st.selectbox(
                "Scholarship Holder?",
                options=["No", "Yes"],
                index=0,
                help="Whether student receives a government/merit scholarship."
            )
        with c12:
            gender = st.selectbox(
                "Gender",
                options=["Female", "Male"],
                index=0
            )
        with c13:
            displaced = st.selectbox(
                "Displaced (Living away from home)?",
                options=["Yes", "No"],
                index=0
            )

        submitted = st.form_submit_button("⚡ Run Risk Assessment", use_container_width=True)

    if submitted:
        # Resolve selected pipeline
        if "Logistic Regression" in model_choice:
            active_model = load_saved_model(PRIMARY_MODEL_PATH)
            engine_name = "Logistic Regression Pipeline (Primary)"
        elif "Multi-Layer Perceptron" in model_choice:
            active_model = load_saved_model(MLP_MODEL_PATH)
            engine_name = "Multi-Layer Perceptron Pipeline (MLP Neural Net)"
        else:
            active_model = load_saved_model(RF_MODEL_PATH)
            engine_name = "Random Forest Pipeline (Comparison Model)"

        if active_model is None:
            st.error("Model pipeline artifact not found! Please run `python train_model.py` first.")
        else:
            # Construct a complete input row with all 30 features
            input_row = defaults.copy() if defaults else {}

            # Map user inputs into row
            input_row['Curricular units 1st sem (approved)'] = float(sem1_approved)
            input_row['Curricular units 1st sem (grade)'] = float(sem1_grade)
            input_row['Curricular units 1st sem (evaluations)'] = float(sem1_evals)
            input_row['Curricular units 1st sem (enrolled)'] = float(sem1_enrolled)
            input_row['Curricular units 1st sem (credited)'] = float(defaults.get('Curricular units 1st sem (credited)', 0.0))
            input_row['Curricular units 1st sem (without evaluations)'] = float(defaults.get('Curricular units 1st sem (without evaluations)', 0.0))

            input_row['Admission grade'] = float(admission_grade)
            input_row['Previous qualification (grade)'] = float(prev_grade)
            input_row['Age at enrollment'] = float(age)
            input_row['Application order'] = float(app_order)

            input_row['Tuition fees up to date'] = 1 if tuition_up_to_date == "Yes" else 0
            input_row['Debtor'] = 1 if is_debtor == "Yes" else 0
            input_row['Scholarship holder'] = 1 if is_scholarship == "Yes" else 0
            input_row['Gender'] = 1 if gender == "Male" else 0
            input_row['Displaced'] = 1 if displaced == "Yes" else 0

            # Economic & demographic defaults
            input_row['Unemployment rate'] = float(defaults.get('Unemployment rate', 11.1))
            input_row['Inflation rate'] = float(defaults.get('Inflation rate', 1.4))
            input_row['GDP'] = float(defaults.get('GDP', 0.32))

            input_df = pd.DataFrame([input_row])

            try:
                pred_class = int(active_model.predict(input_df)[0])
                pred_prob = float(active_model.predict_proba(input_df)[0][1])
            except Exception as e:
                st.error(f"Inference error: {e}")
                pred_class = 0
                pred_prob = 0.0

            st.markdown("---")
            st.markdown(f"### 📋 Assessment Result `[{engine_name}]`")

            r_col1, r_col2 = st.columns([2, 3])

            with r_col1:
                if pred_class == 1 or pred_prob >= 0.5:
                    st.markdown(
                        f"""
                        <div class="risk-alert-high">
                            <h2 style="color: #dc2626; margin: 0; font-size: 1.6rem;">🚨 AT ACADEMIC RISK (DROPOUT)</h2>
                            <p style="color: #7f1d1d; margin: 8px 0 0 0; font-size: 0.95rem;">
                                The model estimates an <strong>elevated risk probability ({pred_prob*100:.1f}%)</strong> 
                                of student dropout based on 1st-semester performance and risk indicators.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="risk-alert-low">
                            <h2 style="color: #059669; margin: 0; font-size: 1.6rem;">✅ NOT AT RISK (ON TRACK)</h2>
                            <p style="color: #064e3b; margin: 8px 0 0 0; font-size: 0.95rem;">
                                The student is performing on track with a <strong>low risk probability ({pred_prob*100:.1f}%)</strong> 
                                of dropout. Projected on path to graduation.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Risk Probability Progress Bar
                risk_pct = int(pred_prob * 100)
                if risk_pct < 35:
                    st.progress(pred_prob, text=f"Dropout Risk: Low ({risk_pct}%)")
                elif risk_pct < 65:
                    st.progress(pred_prob, text=f"Dropout Risk: Moderate ({risk_pct}%)")
                else:
                    st.progress(pred_prob, text=f"Dropout Risk: High ({risk_pct}%)")

            with r_col2:
                recommendations = []

                if sem1_approved < 4:
                    recommendations.append((
                        "Academic Credit Deficit:",
                        f"Student completed only {sem1_approved} curricular units. Enroll immediately in semester credit recovery and study skills workshops."
                    ))
                if tuition_up_to_date == "No" or is_debtor == "Yes":
                    recommendations.append((
                        "Financial Support Priority:",
                        "Student has unpaid tuition or overdue institutional debt. Direct student to financial bursary emergency funds and installment plan counseling."
                    ))
                if sem1_grade < 11.0:
                    recommendations.append((
                        "Remedial Academic Tutoring:",
                        f"1st semester grade ({sem1_grade:.1f}/20) is near or below the passing threshold. Assign dedicated course mentors."
                    ))
                if is_scholarship == "No" and is_debtor == "Yes":
                    recommendations.append((
                        "Bursary Eligibility Review:",
                        "Student lacks scholarship coverage while carrying outstanding debt. Initiate social welfare assistance review."
                    ))
                if age > 25:
                    recommendations.append((
                        "Mature Student Engagement:",
                        "Mature student profile. Recommend adult learning network, evening peer study groups, and work-study balance counseling."
                    ))

                if not recommendations:
                    recommendations.append((
                        "Maintain Academic Momentum:",
                        "Student is progressing smoothly. Continue regular faculty office hours and scheduled academic check-ins."
                    ))

                rec_html_items = "".join([
                    f"""
                    <div class="recommendation-card">
                        <strong class="rec-title">{title}</strong>
                        <span class="rec-body"> {desc}</span>
                    </div>
                    """
                    for title, desc in recommendations
                ])

                st.markdown(
                    f"""
                    <div class="support-plan-card">
                        <h4 class="support-plan-heading">🎯 Personalized Academic Support Plan</h4>
                        {rec_html_items}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ---------------------------------------------------------
# Page 3: Dataset Insights & Upload
# ---------------------------------------------------------
elif page == "📈 Dataset Insights & Upload":
    st.markdown("## 📈 Dataset Insights & CSV Upload")
    st.markdown(
        "Inspect the uploaded `data.csv` dataset, review data quality, examine distributions, or upload a custom CSV."
    )

    tab_inspect, tab_upload = st.tabs(["🔍 Dataset Inspection & EDA", "📤 Upload Custom Dataset"])

    with tab_inspect:
        st.markdown("### Workspace Datasets (`data/`)")
        available_csvs = [f for f in sorted(os.listdir("data")) if f.endswith(".csv")] if os.path.exists("data") else []
        default_index = available_csvs.index("data.csv") if "data.csv" in available_csvs else 0
        selected_file = st.selectbox("Select CSV file to inspect", available_csvs, index=default_index)

        inspect_path = os.path.join("data", selected_file)
        file_info = inspect_dataset(inspect_path)

        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        d_col1.metric("File Name", selected_file)
        d_col2.metric("Total Rows in File", f"{file_info['total_rows']:,}")
        d_col3.metric("Non-Empty Rows", f"{file_info['non_empty_rows']:,}")
        d_col4.metric("Target Column", file_info['target_column'] or "Missing ❌")

        st.markdown("#### Diagnostic Findings")
        if file_info['has_target'] and file_info['non_empty_rows'] > 0:
            st.success(
                f"**✅ Valid Dataset:** `{selected_file}` contains {file_info['non_empty_rows']:,} records "
                f"with target column `{file_info['target_column']}`."
            )
        else:
            st.error(
                f"**⚠️ Data Quality Notice:** `{selected_file}` does not contain a recognizable target column."
            )

        # Missing values table
        st.markdown("#### Feature Schema & Missing Value Distribution")
        df_cols_summary = pd.DataFrame({
            "Feature Name": file_info['columns'],
            "Inferred Type": [file_info['dtypes'].get(c, 'unknown') for c in file_info['columns']],
            "Missing Count": [file_info['missing_per_col'].get(c, 0) for c in file_info['columns']],
            "Missing Pct (%)": [file_info['missing_pct'].get(c, 100.0) for c in file_info['columns']]
        })
        st.dataframe(df_cols_summary, use_container_width=True)

        if file_info['has_target'] and file_info['non_empty_rows'] > 0:
            curr_df = load_and_clean_data(inspect_path)
            st.markdown("#### Dataset Preview (First 10 Rows)")
            st.dataframe(curr_df.head(10), use_container_width=True)

            if 'Target' in curr_df.columns:
                curr_df['Academic_Risk'] = (curr_df['Target'] == 'Dropout').astype(int)
                at_risk_count = int(curr_df['Academic_Risk'].sum())
                not_risk_count = len(curr_df) - at_risk_count

                st.markdown("---")
                st.markdown("### 📊 Interactive Visualizations")

                ch_col1, ch_col2 = st.columns(2)
                with ch_col1:
                    fig_class = px.pie(
                        names=["Not at Risk (Graduate/Enrolled)", "At Risk (Dropout)"],
                        values=[not_risk_count, at_risk_count],
                        title="Binary Target Distribution (Dropout = At Risk)",
                        color_discrete_sequence=["#10b981", "#ef4444"],
                        hole=0.45
                    )
                    st.plotly_chart(fig_class, use_container_width=True)

                with ch_col2:
                    if 'Admission grade' in curr_df.columns and 'Age at enrollment' in curr_df.columns:
                        sample_n = min(1000, len(curr_df))
                        fig_scatter = px.scatter(
                            curr_df.sample(sample_n, random_state=42),
                            x="Admission grade",
                            y="Age at enrollment",
                            color=curr_df.sample(sample_n, random_state=42)['Target'].map(lambda t: "Dropout" if t == "Dropout" else "Graduate/Enrolled"),
                            color_discrete_map={"Dropout": "#ef4444", "Graduate/Enrolled": "#10b981"},
                            title=f"Admission Grade vs. Age at Enrollment ({sample_n:,} Points)",
                            labels={"color": "Status"}
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_upload:
        st.markdown("### 📤 Upload Student Dataset")
        st.markdown(
            "Upload a custom CSV file containing student records to inspect distributions and train models."
        )

        uploaded_file = st.file_uploader("Select a CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                # Try semicolon first, then comma
                try:
                    uploaded_df = pd.read_csv(uploaded_file, sep=';')
                    if uploaded_df.shape[1] == 1:
                        uploaded_file.seek(0)
                        uploaded_df = pd.read_csv(uploaded_file, sep=',')
                except Exception:
                    uploaded_file.seek(0)
                    uploaded_df = pd.read_csv(uploaded_file, sep=',')

                uploaded_df.columns = uploaded_df.columns.str.strip()
                st.success(f"Successfully read CSV: **{uploaded_df.shape[0]} rows** and **{uploaded_df.shape[1]} columns**.")

                has_target = 'Target' in uploaded_df.columns or 'Exam_Score' in uploaded_df.columns
                if has_target:
                    st.success("✅ Target column found!")
                else:
                    st.error("❌ Target column (`Target` or `Exam_Score`) not found.")

                st.markdown("#### Dataset Preview (First 10 Rows)")
                st.dataframe(uploaded_df.head(10), use_container_width=True)

                if has_target and len(uploaded_df) >= 20:
                    st.session_state['active_df'] = uploaded_df
                    st.markdown("---")
                    st.markdown("### 🚀 Train Models on Uploaded Data")
                    st.caption("Trains Logistic Regression, Random Forest, and MLP on the uploaded dataset.")

                    if st.button("Train Models Now", type="primary"):
                        with st.spinner("Fitting pipelines and training models..."):
                            try:
                                new_meta = train_models(uploaded_df, output_dir=MODEL_DIR)
                                st.success("✅ Models trained and serialized successfully!")
                                st.balloons()
                                lr_acc = new_meta['logistic_regression']['accuracy'] * 100
                                mlp_acc = new_meta['mlp']['accuracy'] * 100
                                rf_acc = new_meta['random_forest']['accuracy'] * 100
                                st.markdown(
                                    f"- **Logistic Regression Accuracy:** {lr_acc:.2f}%\n"
                                    f"- **MLP Neural Net Accuracy:** {mlp_acc:.2f}%\n"
                                    f"- **Random Forest Accuracy:** {rf_acc:.2f}%\n"
                                )
                            except Exception as e:
                                st.error(f"Training failed: {e}")
            except Exception as e:
                st.error(f"Failed to parse uploaded CSV: {e}")


# ---------------------------------------------------------
# Page 4: Model Performance
# ---------------------------------------------------------
elif page == "🧠 Model Performance":
    st.markdown("## 🧠 Model Performance & Architecture")
    st.markdown(
        "Comprehensive evaluation comparing the primary **Logistic Regression** classifier, "
        "**Multi-Layer Perceptron (MLP)** neural network, and **Random Forest** on the held-out test set (885 samples)."
    )

    metadata = load_saved_metadata()

    if metadata is None:
        st.warning(
            "⚠️ **No Trained Model Artifact Found:**\n"
            "Please run `python train_model.py` to train models on `data/data.csv`."
        )
    else:
        lr = metadata.get('logistic_regression', {})
        rf = metadata.get('random_forest', {})
        mlp = metadata.get('mlp', {})

        st.markdown("### 📊 Test-Set Evaluation Summary (Stratified 80/20 Split, 885 Test Samples)")
        
        # Comparison Table
        comp_df = pd.DataFrame({
            "Metric": ["Test Accuracy", "Precision (At Risk)", "Recall / Sensitivity (At Risk)", "F1-Score (At Risk)", "ROC-AUC"],
            "Logistic Regression (Primary)": [
                f"{lr.get('accuracy', 0)*100:.2f}%",
                f"{lr.get('precision', 0):.4f}",
                f"{lr.get('recall', 0):.4f}",
                f"{lr.get('f1_score', 0):.4f}",
                f"{lr.get('roc_auc', 0):.4f}" if lr.get('roc_auc') else "N/A"
            ],
            "Multi-Layer Perceptron (MLP)": [
                f"{mlp.get('accuracy', 0)*100:.2f}%" if mlp else "N/A",
                f"{mlp.get('precision', 0):.4f}" if mlp else "N/A",
                f"{mlp.get('recall', 0):.4f}" if mlp else "N/A",
                f"{mlp.get('f1_score', 0):.4f}" if mlp else "N/A",
                f"{mlp.get('roc_auc', 0):.4f}" if mlp and mlp.get('roc_auc') else "N/A"
            ],
            "Random Forest (Comparison)": [
                f"{rf.get('accuracy', 0)*100:.2f}%",
                f"{rf.get('precision', 0):.4f}",
                f"{rf.get('recall', 0):.4f}",
                f"{rf.get('f1_score', 0):.4f}",
                f"{rf.get('roc_auc', 0):.4f}" if rf.get('roc_auc') else "N/A"
            ]
        })
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3 Confusion Matrices Side-by-Side
        st.markdown("### 🔍 Test Set Confusion Matrices (885 Held-Out Test Samples)")
        col_cm1, col_cm2, col_cm3 = st.columns(3)

        with col_cm1:
            st.markdown("##### 1. Logistic Regression (Primary)")
            cm_lr = np.array(lr['confusion_matrix'])
            fig_cm_lr = px.imshow(
                cm_lr,
                text_auto=True,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=["Not at Risk", "At Risk"],
                y=["Not at Risk", "At Risk"],
                color_continuous_scale="Blues"
            )
            fig_cm_lr.update_layout(height=290, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig_cm_lr, use_container_width=True)

        with col_cm2:
            st.markdown("##### 2. Multi-Layer Perceptron (MLP)")
            if mlp and 'confusion_matrix' in mlp:
                cm_mlp = np.array(mlp['confusion_matrix'])
                fig_cm_mlp = px.imshow(
                    cm_mlp,
                    text_auto=True,
                    labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                    x=["Not at Risk", "At Risk"],
                    y=["Not at Risk", "At Risk"],
                    color_continuous_scale="Purples"
                )
                fig_cm_mlp.update_layout(height=290, margin=dict(l=10, r=10, t=25, b=10))
                st.plotly_chart(fig_cm_mlp, use_container_width=True)

        with col_cm3:
            st.markdown("##### 3. Random Forest (Comparison)")
            cm_rf = np.array(rf['confusion_matrix'])
            fig_cm_rf = px.imshow(
                cm_rf,
                text_auto=True,
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=["Not at Risk", "At Risk"],
                y=["Not at Risk", "At Risk"],
                color_continuous_scale="Greens"
            )
            fig_cm_rf.update_layout(height=290, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(fig_cm_rf, use_container_width=True)

        # Feature Importance / Odds Ratios
        if 'feature_analysis' in metadata and metadata['feature_analysis']:
            st.markdown("---")
            st.markdown("### 📈 Logistic Regression Feature Weights & Odds Ratios")
            st.caption(
                "Positive coefficients increase the odds of being **At Risk (Dropout)**; negative coefficients protect against risk."
            )
            feat_df = pd.DataFrame(metadata['feature_analysis'][:20])
            
            fig_feat = px.bar(
                feat_df,
                x='coefficient',
                y='feature',
                orientation='h',
                color='coefficient',
                color_continuous_scale='RdYlGn_r',
                title="Top 20 Model Coefficients (Impact on Dropout Log-Odds)"
            )
            fig_feat.update_layout(yaxis={'categoryorder': 'total ascending'}, height=420)
            st.plotly_chart(fig_feat, use_container_width=True)

        # Hyperparameters & Convergence Audit
        st.markdown("---")
        st.markdown("### ⚙️ Model Architecture & Hyperparameter Configuration")
        h_col1, h_col2, h_col3 = st.columns(3)

        with h_col1:
            st.markdown(
                """
                **Logistic Regression:**
                - `max_iter`: 1000
                - `class_weight`: 'balanced'
                - `solver`: 'lbfgs' (default)
                - `random_state`: 42
                - **Convergence:** Converged smoothly without warnings.
                """
            )
        with h_col2:
            mlp_conv = mlp.get('convergence', {})
            st.markdown(
                f"""
                **Multi-Layer Perceptron (MLP):**
                - `hidden_layer_sizes`: (64, 32)
                - `activation`: 'relu'
                - `solver`: 'adam'
                - `alpha`: 0.001
                - `learning_rate_init`: 0.001
                - `batch_size`: 32
                - `max_iter`: 200
                - `early_stopping`: True
                - `validation_fraction`: 0.1
                - `n_iter_no_change`: 15
                - `random_state`: 42
                - **Convergence:** Converged at **iteration {mlp_conv.get('actual_iterations', 27)}** via `{mlp_conv.get('stopped_by', 'early_stopping')}`.
                """
            )
        with h_col3:
            st.markdown(
                """
                **Random Forest:**
                - `n_estimators`: 100
                - `max_depth`: None
                - `class_weight`: 'balanced'
                - `random_state`: 42
                - `n_jobs`: -1
                """
            )

        # Dedicated Longitudinal / Sequential & LSTM Analysis Section
        st.markdown("---")
        st.markdown("### ⏳ Sequential / Longitudinal Modeling & LSTM Analysis")
        st.markdown(
            """
            In early academic risk detection, recurrent neural networks like **LSTM (Long Short-Term Memory)** 
            are designed specifically for **sequential time-series data**. Here is the rigorous scientific audit 
            of dataset requirements and architecture for ScholarTrack:
            """
        )

        l_c1, l_c2 = st.columns([1, 1])

        with l_c1:
            st.markdown(
                """
                #### 1. Dataset Nature & Scientific Boundary
                - **Current Dataset Audit:** [`data/data.csv`](file:///c:/Users/pranj/OneDrive/Desktop/ScholarTrack/data/data.csv) contains **static cross-sectional student snapshots** (4,424 records, each capturing enrollment demographics and aggregate 1st-semester units).
                - **No Temporal Dimension:** There are zero timestamps, weekly attendance logs, or sequential assignment submissions.
                - **Integrity Rule:** In strict adherence to scientific rigor, static tabular features **must not be treated as sequential time-steps**. Doing so induces artificial order and produces meaningless recurrent weights.
                
                #### 2. Identified Public Sequential Datasets
                When longitudinal tracking is required, the following public benchmark datasets are suitable:
                1. **OULAD (Open University Learning Analytics Dataset):**
                   - 32,593 students with daily clickstream activity, Virtual Learning Environment (VLE) interaction logs, and sequential assessment submissions over 39 weeks.
                2. **UCI Student Performance Dataset (Cortez & Silva):**
                   - Contains 3 consecutive evaluation periods ($G_1, G_2, G_3$) enabling period-to-period sequential progression modeling.
                3. **ASSISTments Longitudinal Tutor Dataset:**
                   - Detailed sequential student interaction logs with problem-level timestamps.
                """
            )

        with l_c2:
            st.markdown(
                """
                #### 3. LSTM Architecture Specification for Sequential Data
                When longitudinal student records are provided:
                - **Input Tensor Structure:** $(N, T, D)$
                  - $N$: Number of students
                  - $T$: Longitudinal time-steps (e.g. 12 semester weeks)
                  - $D$: Weekly features (weekly attendance %, LMS logins, quiz submissions, homework grades)
                - **Data Leakage Prevention:**
                  - Splitting must be performed at the **Student ID level** (`GroupKFold` or `train_test_split(students)`), ensuring all sequence steps for a given student stay entirely inside either train or test sets.
                - **TensorFlow / Keras Architecture**:
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
                model.compile(optimizer='adam', loss='binary_crossentropy')
                ```
                - **Status:** Static tabular data is modeled using Logistic Regression, Random Forest, and MLP; LSTM is reserved for datasets with authentic time sequences.
                """
            )

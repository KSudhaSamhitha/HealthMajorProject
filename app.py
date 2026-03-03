import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(
    page_title="CardioXAI",
    layout="wide",
    page_icon="❤️"
)

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0f172a;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-163ttbj, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e293b);
    }
    
    /* Sidebar text */
    .sidebar .sidebar-content {
        color: white;
    }
    
    /* Button styling for sidebar menu */
    .stButton button {
        background-color: transparent;
        color: #94a3b8;
        border: none;
        text-align: left;
        padding: 10px 15px;
        margin: 2px 0;
        border-radius: 8px;
        transition: all 0.3s ease;
        font-size: 16px;
    }
    
    .stButton button:hover {
        background-color: #334155;
        color: white;
        border: none;
    }
    
    .stButton button:focus {
        background: linear-gradient(90deg, #312e81, #1e293b);
        color: white;
        border-left: 3px solid #60a5fa;
        border-radius: 8px 0 0 8px;
    }
    
    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #1e293b, #312e81);
        padding: 40px;
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
        border: 1px solid #334155;
    }
    
    .hero h4 {
        color: #94a3b8;
        font-size: 14px;
        letter-spacing: 1px;
        margin-bottom: 10px;
        font-weight: 500;
    }
    
    .hero h1 {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 20px;
        line-height: 1.2;
        color: white;
    }
    
    .hero p {
        color: #cbd5e1;
        font-size: 16px;
        max-width: 800px;
        line-height: 1.6;
    }
    
    /* Chips/tags */
    .chip-container {
        margin: 30px 0;
        display: flex;
        flex-wrap: wrap;
    }
    
    .chip {
        display: inline-block;
        padding: 8px 20px;
        margin: 5px 8px 5px 0;
        border-radius: 30px;
        background-color: #334155;
        color: white;
        font-size: 15px;
        font-weight: 500;
        border: 1px solid #475569;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #2d3748);
        padding: 30px 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        border: 1px solid #334155;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #60a5fa;
    }
    
    .metric-card h2 {
        font-size: 42px;
        font-weight: 700;
        margin: 0;
        color: #60a5fa;
    }
    
    .metric-card p {
        color: #94a3b8;
        font-size: 14px;
        margin: 10px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section cards */
    .section-card {
        background: linear-gradient(135deg, #1e293b, #2d3748);
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin: 20px 0;
        border: 1px solid #334155;
    }
    
    .section-card h2 {
        font-size: 28px;
        font-weight: 600;
        margin-bottom: 20px;
        color: white;
        border-bottom: 2px solid #334155;
        padding-bottom: 10px;
    }
    
    .section-card h3 {
        font-size: 20px;
        font-weight: 600;
        color: #60a5fa;
        margin-bottom: 10px;
    }
    
    .section-card p {
        color: #cbd5e1;
        font-size: 16px;
        line-height: 1.6;
    }
    
    /* Step cards */
    .step-card {
        background-color: #2d3748;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #4a5568;
        height: 100%;
        transition: all 0.3s ease;
    }
    
    .step-card:hover {
        border-color: #60a5fa;
        transform: translateY(-3px);
    }
    
    .step-number {
        color: #60a5fa;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .step-title {
        color: white;
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .step-description {
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.6;
    }
    
    /* Dataset stats */
    .dataset-stat {
        background-color: #2d3748;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #4a5568;
        height: 100%;
    }
    
    .dataset-stat h3 {
        font-size: 20px;
        font-weight: 600;
        margin: 0 0 15px 0;
        color: white;
    }
    
    .stat-number {
        color: #60a5fa;
        font-size: 32px;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .stat-label {
        color: #94a3b8;
        font-size: 14px;
        margin: 5px 0;
    }
    
    .stat-desc {
        color: #64748b;
        font-size: 13px;
        margin-top: 10px;
        font-style: italic;
    }
    
    /* Feature grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
        gap: 10px;
        margin: 20px 0;
    }
    
    .feature-item {
        background-color: #2d3748;
        padding: 10px 15px;
        border-radius: 8px;
        color: #e2e8f0;
        font-size: 14px;
        border: 1px solid #4a5568;
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        border-color: #60a5fa;
        background-color: #374151;
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #60a5fa, transparent);
        margin: 40px 0;
    }
    
    /* Model performance item */
    .model-item {
        display: flex;
        align-items: center;
        padding: 12px 15px;
        background-color: #2d3748;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #4a5568;
    }
    
    .model-name {
        color: white;
        font-weight: 500;
        flex: 1;
    }
    
    .model-badge {
        background-color: #60a5fa;
        color: #0f172a;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 10px;
    }
    
    .best-model {
        border-color: #60a5fa;
        background-color: #374151;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 14px;
        margin-top: 50px;
        padding: 20px 0;
        border-top: 1px solid #334155;
    }
    
    /* Navigation menu */
    .nav-menu {
        color: white;
        padding: 10px;
    }
    
    /* Class distribution */
    .dist-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #334155;
    }
    
    .dist-label {
        color: #94a3b8;
    }
    
    .dist-value {
        color: white;
        font-weight: 600;
    }
    
    .dist-percent {
        color: #60a5fa;
        font-weight: 500;
    }
    
    /* Snapshot container */
    .snapshot-container {
        background: linear-gradient(135deg, #1e293b, #334155);
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
    }
    
    /* Prediction result card */
    .prediction-card {
        background: linear-gradient(135deg, #1e293b, #2d3748);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #334155;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.markdown("## ❤️ CardioXAI")
    st.markdown("### Heart Risk Prediction")
    st.markdown("---")
    
    # Initialize session state for menu if not exists
    if "menu" not in st.session_state:
        st.session_state.menu = "Home"
    
    # Create styled buttons for menu
    col1, col2 = st.columns([0.1, 0.9])
    with col2:
        if st.button("🏠 Home", key="btn_home", use_container_width=True):
            st.session_state.menu = "Home"
        if st.button("📊 Risk Prediction", key="btn_risk", use_container_width=True):
            st.session_state.menu = "Risk Prediction"
        if st.button("📈 Model Comparison", key="btn_compare", use_container_width=True):
            st.session_state.menu = "Model Comparison"
        if st.button("🔍 Explainability", key="btn_explain", use_container_width=True):
            st.session_state.menu = "Explainability"
    
    # Get current menu from session state
    menu = st.session_state.menu
    
    st.markdown("---")
    st.markdown("<p style='color: #64748b; font-size: 12px; text-align: center;'>Major Project — Final Year<br>Multimodal XAI · 2024</p>", unsafe_allow_html=True)

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    try:
        X = pd.read_csv("processed_data/X_scaled.csv")
        y = pd.read_csv("processed_data/y.csv")
        return X, y
    except:
        # Return dummy data if files don't exist yet
        X = pd.DataFrame({
            'age': range(1000),
            'sex': [1] * 1000,
            'chest_pain_type': [0] * 1000,
        })
        y = pd.DataFrame({'target': [0] * 840 + [1] * 160})
        return X, y

X, y = load_data()

# -------------------------
# HOME PAGE
# -------------------------
if menu == "Home":
    
    # Hero Section
    st.markdown("""
    <div class="hero">
        <h4>MAJOR PROJECT · FINAL YEAR</h4>
        <h1>Multimodal Explainable AI<br>for Heart Disease Risk Prediction</h1>
        <p>
        A comprehensive machine learning system that fuses UCI Heart Disease 
        and Framingham datasets to predict cardiovascular risk with SHAP & LIME.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model Chips
    st.markdown("""
    <div class="chip-container">
        <span class="chip">XGBoost</span>
        <span class="chip">Random Forest</span>
        <span class="chip">Gradient Boosting</span>
        <span class="chip">Logistic Regression</span>
        <span class="chip">SHAP</span>
        <span class="chip">LIME</span>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>5,263</h2>
            <p>Total samples</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>26</h2>
            <p>Fused features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>5</h2>
            <p>ML models trained</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h2>~88%</h2>
            <p>Best accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ML Pipeline Overview Section
    st.markdown("""
    <div class="section-card">
        <h2>ML Pipeline Overview</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Pipeline Steps
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">STEP 01</div>
            <div class="step-title"><strong>Data Fusion</strong></div>
            <div class="step-description">Merge UCI & Framingham datasets with feature alignment</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">STEP 02</div>
            <div class="step-title"><strong>Preprocessing</strong></div>
            <div class="step-description">Handle missing values, encode categories, StandardScaler</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">STEP 03</div>
            <div class="step-title"><strong>Model Training</strong></div>
            <div class="step-description">Train 5 models: LR, RF, GB, XGBoost, Tuned XGBoost</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="step-card">
            <div class="step-number">STEP 04</div>
            <div class="step-title"><strong>Explainability</strong></div>
            <div class="step-description">SHAP TreeExplainer + LIME local explanations</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Model Performance Snapshot
    st.markdown('<div class="snapshot-container">', unsafe_allow_html=True)
    st.markdown('<h2 style="color:white; margin-bottom:20px;">Model Performance Snapshot</h2>', unsafe_allow_html=True)

    # Simplified model list as shown in image 1
    st.markdown("""
    <div style="margin-bottom: 30px;">
        <div class="model-item best-model" style="justify-content: flex-start;">
            <span style="color: #facc15; margin-right: 10px;">✓</span>
            <span class="model-name"><strong>Tuned XGBoost</strong> (Best Model)</span>
        </div>
        <div class="model-item" style="justify-content: flex-start;">
            <span style="color: #60a5fa; margin-right: 10px;">•</span>
            <span class="model-name">Random Forest</span>
        </div>
        <div class="model-item" style="justify-content: flex-start;">
            <span style="color: #60a5fa; margin-right: 10px;">•</span>
            <span class="model-name">Gradient Boosting</span>
        </div>
    </div>

    <div style="text-align: center; margin-top: 20px;">
        <div style="font-size: 48px; font-weight: 700; color: #60a5fa;">88%</div>
        <p style="color: #94a3b8;">Best model accuracy</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Dataset Fusion Section
    st.markdown("""
    <div class="section-card">
        <h2>Multimodal Dataset Fusion</h2>
        <p>
        This project merges two independent cardiovascular datasets to create a richer, 
        more comprehensive feature space. By combining clinical examination data from the 
        UCI Heart Disease dataset with longitudinal epidemiological data from the Framingham 
        Heart Study, the model captures both immediate clinical markers and long-term 
        lifestyle risk factors.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="dataset-stat">
            <h3>UCI Heart Disease</h3>
            <div class="stat-number">1,025</div>
            <div class="stat-label">Records · 13 Features</div>
            <div class="stat-desc">Clinical examination data from cardiology patients</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="dataset-stat">
            <h3>Framingham Heart Study</h3>
            <div class="stat-number">4,238</div>
            <div class="stat-label">Records · 15 Features</div>
            <div class="stat-desc">Longitudinal epidemiological cohort study</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="dataset-stat">
            <h3>Fused Dataset</h3>
            <div class="stat-number">5,263</div>
            <div class="stat-label">Records · 26 Features</div>
            <div class="stat-desc">Combined, preprocessed & scaled</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Features Section
    st.markdown("""
    <div class="section-card">
        <h2>26 Fused Features</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display features in a grid
    features = [
        "age", "sex", "chest_pain_type", "resting_bp", "cholesterol", 
        "fasting_blood_sugar", "resting_ecg", "max_heart_rate", "exercise_angina",
        "st_depression", "st_slope", "num_vessels", "thalassemia", "education",
        "current_smoker", "cigs_per_day", "bp_meds", "prevalent_stroke",
        "prevalent_hypertension", "diabetes", "total_cholesterol", "sys_bp",
        "dia_bp", "bmi", "heart_rate", "glucose"
    ]
    
    # Create feature grid
    feature_html = '<div class="feature-grid">'
    for feature in features:
        feature_html += f'<div class="feature-item">{feature}</div>'
    feature_html += '</div>'
    
    st.markdown(feature_html, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Class Distribution Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="section-card">
            <h2>Class Distribution</h2>
            <p style="color: #94a3b8; margin-bottom: 15px;"><strong>Fused Dataset (after preprocessing)</strong></p>
            <div class="dist-item">
                <span class="dist-label">No Heart Disease (0)</span>
                <span><span class="dist-value">4,401</span> <span class="dist-percent">~84%</span></span>
            </div>
            <div class="dist-item">
                <span class="dist-label">Heart Disease (1)</span>
                <span><span class="dist-value">862</span> <span class="dist-percent">~16%</span></span>
            </div>
            <p style="color: #64748b; font-size: 14px; margin-top: 15px; font-style: italic;">Class imbalance addressed during model training</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="section-card">
            <h2>Individual Datasets</h2>
            <p style="color: #94a3b8; margin-bottom: 15px;"><strong>UCI Heart</strong></p>
            <div class="dist-item">
                <span class="dist-label">No Disease</span>
                <span class="dist-value">500</span>
            </div>
            <div class="dist-item">
                <span class="dist-label">Disease</span>
                <span class="dist-value">525</span>
            </div>
            <p style="color: #94a3b8; margin: 20px 0 15px 0;"><strong>Framingham</strong></p>
            <div class="dist-item">
                <span class="dist-label">No Disease</span>
                <span class="dist-value">3,596</span>
            </div>
            <div class="dist-item">
                <span class="dist-label">Disease</span>
                <span class="dist-value">642</span>
            </div>
            <p style="color: #64748b; font-size: 14px; margin-top: 15px; font-style: italic;">Heart dataset is nearly balanced; Framingham is imbalanced ~85:15</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer with accuracy
    st.markdown("""
    <div class="footer">
        Major Project — Final Year · Multimodal XAI - 2024 · <span style="color: #60a5fa; font-weight: 600;">88% Best model accuracy</span>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# RISK PREDICTION PAGE
# -------------------------
elif menu == "Risk Prediction":

    # ===========================
    # LOAD MODEL
    # ===========================
    model_path = "outputs/results/Tuned_XGBoost.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        model_loaded = True
    else:
        model_loaded = False
        st.warning("⚠️ Model file not found. Please train the model first.")

    st.markdown("""
    <div class="hero">
        <h4>Powered by Tuned XGBoost</h4>
        <h1>Heart Disease Risk Prediction</h1>
        <p>Enter patient clinical and lifestyle data to estimate cardiovascular risk probability</p>
    </div>
    """, unsafe_allow_html=True)

    # Create two columns for main layout
    col_left, col_right = st.columns([2, 1])

    with col_left:
        
        # ===========================
        # DECISION THRESHOLD
        # ===========================
        st.markdown("### Decision Threshold")
        st.markdown("*Adjust threshold to balance sensitivity vs specificity. Lower = more sensitive to risk.*")
        threshold = st.slider("", 0.0, 1.0, 0.50, 0.01, key="threshold", label_visibility="collapsed")
        
        st.markdown("---")
        
        # ===========================
        # DEMOGRAPHICS
        # ===========================
        st.subheader("Demographics")
        
        col_demo1, col_demo2, col_demo3 = st.columns(3)
        with col_demo1:
            age = st.number_input("Age", min_value=20, max_value=90, value=55, step=1)
        with col_demo2:
            sex = st.selectbox("Sex", ["Male (1)", "Female (0)"])
        with col_demo3:
            education = st.selectbox("Education Level", [
                "High School/GED", "Some College", "Bachelor", "Master", "PhD"
            ])
        
        col_demo4, col_demo5 = st.columns(2)
        with col_demo4:
            bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.5, step=0.1)
        with col_demo5:
            pass  # Empty for spacing
        
        st.markdown("---")
        
        # ===========================
        # CLINICAL FEATURES
        # ===========================
        st.subheader("Clinical Features")
        
        col_clin1, col_clin2 = st.columns(2)
        with col_clin1:
            chest_pain = st.selectbox("Chest Pain Type", [
                "Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"
            ])
            resting_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=220, value=130, step=1)
            cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=245, step=1)
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])
            resting_ecg = st.selectbox("Resting ECG", [
                "Normal", "ST-T Abnormality", "LV Hypertrophy"
            ])
        
        with col_clin2:
            max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150, step=1)
            exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
            st_depression = st.number_input("ST Depression", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
            st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
            num_vessels = st.number_input("Number of Major Vessels", min_value=0, max_value=4, value=0, step=1)
            thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        
        st.markdown("---")
        
        # ===========================
        # ADDITIONAL CLINICAL
        # ===========================
        st.subheader("Additional Clinical Measurements")
        
        col_add1, col_add2, col_add3 = st.columns(3)
        with col_add1:
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=300, value=130, step=1)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=50, max_value=150, value=82, step=1)
        
        with col_add2:
            total_chol = st.number_input("Total Cholesterol (mg/dl)", min_value=100, max_value=700, value=235, step=1)
            glucose = st.number_input("Glucose (mg/dl)", min_value=40, max_value=400, value=80, step=1)
        
        with col_add3:
            heart_rate = st.number_input("Heart Rate (Framingham)", min_value=40, max_value=150, value=75, step=1)
        
        st.markdown("---")
        
        # ===========================
        # LIFESTYLE & MEDICAL HISTORY
        # ===========================
        st.subheader("Lifestyle & Medical History")
        
        col_life1, col_life2, col_life3 = st.columns(3)
        with col_life1:
            smoker = st.selectbox("Current Smoker", ["No", "Yes"])
            cigs_per_day = st.number_input("Cigarettes Per Day", min_value=0, max_value=70, value=0, step=1)
        
        with col_life2:
            bp_meds = st.selectbox("Blood Pressure Medications", ["No", "Yes"])
            stroke = st.selectbox("Prevalent Stroke", ["No", "Yes"])
        
        with col_life3:
            hypertension = st.selectbox("Prevalent Hypertension", ["No", "Yes"])
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        
        st.markdown("---")
        
        # Predict button
        predict_button = st.button("💙 Predict Risk", use_container_width=True)

    # ===========================
    # RIGHT PANEL - PREDICTION RESULT
    # ===========================
    with col_right:
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b, #2d3748); padding: 25px; border-radius: 15px; border: 1px solid #334155;">
            <h3 style="color: white; margin-top: 0;">Prediction Result</h3>
        """, unsafe_allow_html=True)
        
        if predict_button and model_loaded:
            
            # Convert categorical inputs to numeric
            sex_val = 1 if "Male" in sex else 0
            fasting_val = 1 if "Yes" in fasting_bs else 0
            exercise_val = 1 if exercise_angina == "Yes" else 0
            smoker_val = 1 if smoker == "Yes" else 0
            bp_meds_val = 1 if bp_meds == "Yes" else 0
            stroke_val = 1 if stroke == "Yes" else 0
            hyper_val = 1 if hypertension == "Yes" else 0
            diabetes_val = 1 if diabetes == "Yes" else 0
            
            # Map categorical to numeric values
            chest_pain_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal": 2, "Asymptomatic": 3}
            ecg_map = {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2}
            slope_map = {"Up": 0, "Flat": 1, "Down": 2}
            thal_map = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2}
            edu_map = {"High School/GED": 0, "Some College": 1, "Bachelor": 2, "Master": 3, "PhD": 4}
            
            chest_pain_val = chest_pain_map[chest_pain]
            ecg_val = ecg_map[resting_ecg]
            slope_val = slope_map[st_slope]
            thal_val = thal_map[thalassemia]
            edu_val = edu_map[education]
            
            # Create input array - match your X_scaled.csv column order
            # IMPORTANT: Adjust this order to match your actual feature columns
            input_data = np.array([[
                age, sex_val, chest_pain_val, resting_bp, cholesterol,
                fasting_val, ecg_val, max_hr, exercise_val,
                st_depression, slope_val, num_vessels, thal_val, edu_val,
                smoker_val, cigs_per_day, bp_meds_val, stroke_val,
                hyper_val, diabetes_val, total_chol, systolic_bp,
                diastolic_bp, bmi, heart_rate, glucose
            ]])
            
            # Get prediction probability
            prob = model.predict_proba(input_data)[0][1]
            
            # Display result based on threshold
            if prob >= threshold:
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.2); border: 2px solid #ef4444; border-radius: 10px; padding: 20px; margin: 15px 0; text-align: center;">
                    <h4 style="color: #ef4444; margin: 0;">⚠ HIGH RISK</h4>
                    <p style="color: white; font-size: 36px; font-weight: 700; margin: 10px 0;">{prob:.1%}</p>
                    <p style="color: #94a3b8;">Probability of heart disease</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: rgba(34, 197, 94, 0.2); border: 2px solid #22c55e; border-radius: 10px; padding: 20px; margin: 15px 0; text-align: center;">
                    <h4 style="color: #22c55e; margin: 0;">✅ LOW RISK</h4>
                    <p style="color: white; font-size: 36px; font-weight: 700; margin: 10px 0;">{prob:.1%}</p>
                    <p style="color: #94a3b8;">Probability of heart disease</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show threshold info
            st.markdown(f"""
            <div style="background-color: #2d3748; padding: 15px; border-radius: 8px; margin-top: 15px;">
                <p style="color: #94a3b8; margin: 0;">Current threshold: <span style="color: white; font-weight: 600;">{threshold:.2f}</span></p>
                <p style="color: #94a3b8; margin: 5px 0 0 0;">Classification: <span style="color: white;">{"High Risk" if prob >= threshold else "Low Risk"}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
        elif not model_loaded:
            st.markdown("""
            <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 10px; padding: 20px; margin: 15px 0; text-align: center;">
                <p style="color: #ef4444;">⚠ Model not loaded</p>
                <p style="color: #94a3b8; font-size: 14px;">Please train the model first</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #2d3748; border: 1px solid #4a5568; border-radius: 10px; padding: 40px 20px; margin: 15px 0; text-align: center;">
                <p style="color: #94a3b8; font-size: 16px;">📋 Fill in patient details</p>
                <p style="color: #64748b; font-size: 14px;">and click Predict Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional info footer
        st.markdown("""
        <div style="margin-top: 30px; padding: 15px; background-color: #1e293b; border-radius: 10px;">
            <p style="color: #64748b; font-size: 12px; text-align: center; margin: 0;">
                Major Project — Final Year<br>
                Multimodal XAI · 2024<br>
                <span style="color: #60a5fa; font-weight: 600;">Best model accuracy: 88%</span>
            </p>
        </div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# -------------------------
# MODEL COMPARISON PAGE
# -------------------------
elif menu == "Model Comparison":

    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.markdown("""
    <div class="hero">
        <h4>5 Models Evaluated</h4>
        <h1>Model Performance Comparison</h1>
        <p>Evaluation metrics across all trained classifiers on the held-out test set</p>
    </div>
    """, unsafe_allow_html=True)

    # ===========================
    # LOAD METRICS CSV
    # ===========================
    try:
        df = pd.read_csv("outputs/results/model_comparison.csv")
        
        # Clean column names - remove any spaces and standardize
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
        
        # Define metric mappings to handle different possible column names
        metric_mapping = {
            'Accuracy': ['Accuracy', 'accuracy', 'ACCURACY'],
            'Precision': ['Precision', 'precision', 'PRECISION'],
            'Recall': ['Recall', 'recall', 'RECALL'],
            'F1_Score': ['F1 Score', 'F1_Score', 'f1_score', 'F1', 'f1'],
            'ROC_AUC': ['ROC AUC', 'ROC_AUC', 'roc_auc', 'AUC', 'auc']
        }
        
        # Find actual column names in the dataframe
        actual_columns = {}
        for standard_name, possible_names in metric_mapping.items():
            for col in df.columns:
                if col in possible_names or col.replace('_', ' ') in possible_names:
                    actual_columns[standard_name] = col
                    break
        
        # Rename columns to standard names
        df = df.rename(columns={v: k for k, v in actual_columns.items()})
        
        # Ensure Model column exists
        if 'Model' not in df.columns:
            # Try to find model column
            for col in df.columns:
                if 'model' in col.lower():
                    df = df.rename(columns={col: 'Model'})
                    break
        
        # Display raw data for debugging (optional - remove in production)
        with st.expander("Debug: View Raw Data"):
            st.write("Column names:", df.columns.tolist())
            st.dataframe(df)
        
        # Identify best model using ROC AUC
        if 'ROC_AUC' in df.columns:
            best_row = df.loc[df['ROC_AUC'].idxmax()]
            best_model = best_row['Model']
            
            # Get metrics with safe access
            accuracy = best_row.get('Accuracy', 0)
            f1 = best_row.get('F1_Score', 0)
            roc_auc = best_row.get('ROC_AUC', 0)
            
            # ===========================
            # BEST MODEL CARD
            # ===========================
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #312e81, #1e293b);
                padding: 30px;
                border-radius: 20px;
                border: 2px solid #facc15;
                margin-bottom: 30px;
                color: white;
            ">
                <h4 style="color: #facc15; margin: 0 0 10px 0;">🏆 BEST PERFORMING MODEL</h4>
                <h2 style="color: white; margin: 0 0 15px 0; font-size: 36px;">{best_model}</h2>
                <div style="display: flex; gap: 30px; flex-wrap: wrap;">
                    <div><span style="color: #94a3b8;">Accuracy:</span> <span style="color: #facc15; font-weight: 700;">{accuracy:.1f}%</span></div>
                    <div><span style="color: #94a3b8;">F1 Score:</span> <span style="color: #facc15; font-weight: 700;">{f1:.1f}%</span></div>
                    <div><span style="color: #94a3b8;">ROC AUC:</span> <span style="color: #facc15; font-weight: 700;">{roc_auc:.1f}%</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("ROC AUC column not found in the data. Please check your CSV file.")
            
    except FileNotFoundError:
        st.error("⚠️ Model comparison file not found. Please run training first to generate metrics.")
        # Create dummy data for demonstration
        data = {
            'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'Tuned XGBoost'],
            'Accuracy': [83.7, 85.7, 86.3, 87.4, 88.6],
            'Precision': [56.9, 62.5, 64.3, 66.7, 70.6],
            'Recall': [41.7, 45.8, 47.9, 50.0, 54.2],
            'F1_Score': [48.1, 52.9, 54.9, 57.1, 61.3],
            'ROC_AUC': [87.1, 89.2, 90.4, 91.9, 93.1]
        }
        df = pd.DataFrame(data)
        best_row = df.loc[df['ROC_AUC'].idxmax()]
        best_model = best_row['Model']

    # ===========================
    # METRICS TABLE
    # ===========================
    st.markdown("### Detailed Metrics Table")
    st.markdown("*Overall correct predictions | True positives / predicted positives | True positives / actual positives | Harmonic mean of precision and recall | Area under the ROC curve*")

    # Create styled dataframe for display
    display_df = df.copy()
    
    # Format percentages and add stars for best values
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']:
        if metric in display_df.columns:
            max_val = display_df[metric].max()
            display_df[metric] = display_df[metric].apply(
                lambda x: f"⭐ {x:.1f}%" if x == max_val else f"{x:.1f}%"
            )

    # Rename columns for display
    display_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    st.dataframe(display_df, use_container_width=True)

    # ===========================
    # METRIC CARDS SECTION (Vertical Layout)
    # ===========================
    st.markdown("### Metric Breakdown")
    
    # Define colors for each metric
    metric_colors = {
        'Accuracy': '#60a5fa',  # Blue
        'Precision': '#8b5cf6',  # Purple
        'Recall': '#ec4899',     # Pink
        'F1_Score': '#f59e0b',   # Orange
        'ROC_AUC': '#10b981'     # Green
    }
    
    metric_display_names = {
        'Accuracy': 'Accuracy',
        'Precision': 'Precision',
        'Recall': 'Recall',
        'F1_Score': 'F1 Score',
        'ROC_AUC': 'ROC AUC'
    }
    
    # Create 5 rows, one for each metric
    for metric, color in metric_colors.items():
        if metric in df.columns:
            st.markdown(f"#### {metric_display_names[metric]}")
            
            # Sort by this metric descending
            sorted_df = df.sort_values(metric, ascending=False)
            
            for _, row in sorted_df.iterrows():
                value = row[metric]
                model_name = row['Model']
                
                # Determine if this is the best model for this metric
                is_best = value == df[metric].max()
                star = "⭐ " if is_best else ""
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**{model_name}**")
                with col2:
                    # Progress bar
                    st.markdown(f"""
                    <div style="margin-top: 8px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                            <span>{star}{value:.1f}%</span>
                        </div>
                        <div style="background-color: #334155; height: 8px; border-radius: 4px;">
                            <div style="width: {value}%; background-color: {color}; height: 8px; border-radius: 4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("---")

    # ===========================
    # VISUAL COMPARISON CHARTS
    # ===========================
    st.markdown("### Visual Comparison")
    
    tab1, tab2 = st.tabs(["📊 Bar Chart", "📈 Radar Chart"])
    
    with tab1:
        # Metric selector for bar chart
        selected_metric = st.selectbox(
            "Select Metric",
            ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC'],
            format_func=lambda x: metric_display_names[x]
        )
        
        # Create bar chart
        fig = px.bar(
            df,
            x='Model',
            y=selected_metric,
            color='Model',
            text=df[selected_metric].round(1),
            title=f'{metric_display_names[selected_metric]} by Model',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        fig.update_traces(
            texttemplate='%{text}%',
            textposition='outside',
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5,
            opacity=0.8
        )
        
        fig.update_layout(
            yaxis_title=f"{metric_display_names[selected_metric]} (%)",
            xaxis_title="",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=500,
            showlegend=False
        )
        
        fig.update_yaxes(gridcolor='#334155', gridwidth=1)
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Radar chart for all metrics
        st.markdown("#### Model Comparison Radar Chart")
        
        # Normalize data for radar chart (0-100 scale)
        radar_df = df.copy()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
        
        fig = go.Figure()
        
        for _, row in radar_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics],
                theta=[metric_display_names[m] for m in metrics],
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor='#334155'
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True,
            legend=dict(
                font_color='white',
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("""
    <div class="footer">
        Major Project — Final Year · Multimodal XAI - 2024 · <span style="color: #60a5fa; font-weight: 600;">Best model accuracy: 88.6%</span>
    </div>
    """, unsafe_allow_html=True)

    # ===========================
    # METRICS TABLE
    # ===========================
    st.subheader("Detailed Metrics Table")

    styled_df = df.copy()

    for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]:
        max_value = df[metric].max()
        styled_df[metric] = styled_df[metric].apply(
            lambda x: f"⭐ {x:.1f}%" if x == max_value else f"{x:.1f}%"
        )

    st.dataframe(styled_df, use_container_width=True)

    # ===========================
    # VISUAL COMPARISON BAR CHART
    # ===========================
    st.subheader("Visual Comparison")

    metric_option = st.radio(
        "Select Metric",
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
        horizontal=True
    )

    fig = px.bar(
        df,
        x="Model",
        y=metric_option,
        color="Model",
        text=df[metric_option].round(1),
    )

    fig.update_layout(
        yaxis_title=f"{metric_option} (%)",
        xaxis_title="Model",
        height=500
    )

    fig.update_traces(texttemplate='%{text}%', textposition='outside')

    st.plotly_chart(fig, use_container_width=True)

    # ===========================
    # METRIC CARDS SECTION
    # ===========================
    st.markdown("### Metric Breakdown")

    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    def metric_card(column, metric_name, color):
        with column:
            st.markdown(f"#### {metric_name}")
            for _, row in df.iterrows():
                value = row[metric_name]
                st.markdown(f"""
                <div style="margin-bottom:12px;">
                    <b>{row['Model']}</b> — {value:.1f}%
                    <div style="
                        background:#eaeaea;
                        height:8px;
                        border-radius:5px;
                        margin-top:4px;">
                        <div style="
                            width:{value}%;
                            height:8px;
                            background:{color};
                            border-radius:5px;">
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    metric_card(col1, "Accuracy", "#6C63FF")
    metric_card(col2, "Precision", "#8E44AD")
    metric_card(col3, "Recall", "#E84393")
    metric_card(col4, "F1 Score", "#F39C12")
    metric_card(col5, "ROC AUC", "#27AE60")
    
# -------------------------
# EXPLAINABILITY PAGE
# -------------------------
elif menu == "Explainability":
    st.markdown("""
    <div class="hero">
        <h4>EXPLAINABILITY</h4>
        <h1>SHAP & LIME Explanations</h1>
        <p>Understand model predictions with state-of-the-art explainability techniques.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Placeholder for explainability
    st.markdown("""
    <div class="section-card">
        <h2>Model Explanations</h2>
        <p style="color: #94a3b8;">SHAP and LIME visualizations will be displayed here.</p>
    </div>
    """, unsafe_allow_html=True)
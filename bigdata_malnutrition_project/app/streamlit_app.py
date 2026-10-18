import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# Set Streamlit Page Configuration with clean title and layout
st.set_page_config(
    page_title="UNICEF Malnutrition & Survival Prediction System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Minimalist Styling (White Background, Red Buttons, High Contrast Black/White Text)
st.markdown("""
<style>
    /* Global Base */
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }

    [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #F3F4F6 !important;
    }

    /* Minimalist Header Section */
    .app-header-container {
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #111827;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.25;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #4B5563;
        margin-top: 0.35rem;
        margin-bottom: 0;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Section Subheadings */
    .section-title {
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #374151;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #E5E7EB;
    }

    /* Form Container */
    [data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
        padding: 1.5rem !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04) !important;
    }

    /* Input & Widget Styling */
    label[data-testid="stWidgetLabel"] p {
        color: #374151 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }

    /* Streamlit Tabs */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #4B5563 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1.25rem !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        transition: color 0.15s ease-in-out, border-color 0.15s ease-in-out !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #111827 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #DC2626 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #DC2626 !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #DC2626 !important;
    }
    div[data-baseweb="tab-border"] {
        background-color: #E5E7EB !important;
    }

    /* Red Primary Buttons */
    div[data-testid="stFormSubmitButton"] {
        margin-top: 1rem !important;
    }
    div[data-testid="stFormSubmitButton"] > button,
    div.stButton > button {
        background-color: #DC2626 !important;
        color: #FFFFFF !important;
        border: 1px solid #DC2626 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.01em !important;
        padding: 0.65rem 1.5rem !important;
        transition: background-color 0.15s ease-in-out, border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        width: 100% !important;
    }
    div[data-testid="stFormSubmitButton"] > button p,
    div.stButton > button p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover,
    div.stButton > button:hover {
        background-color: #B91C1C !important;
        border-color: #B91C1C !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stFormSubmitButton"] > button:active,
    div.stButton > button:active,
    div[data-testid="stFormSubmitButton"] > button:focus,
    div.stButton > button:focus {
        background-color: #991B1B !important;
        border-color: #991B1B !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.25) !important;
        outline: none !important;
    }

    /* Minimalist Clinical Result Cards */
    .clinical-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1.25rem;
        height: 100%;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    .card-border-green {
        border-left: 4px solid #16A34A;
    }
    .card-border-red {
        border-left: 4px solid #DC2626;
    }
    .card-border-amber {
        border-left: 4px solid #D97706;
    }
    .card-border-gray {
        border-left: 4px solid #4B5563;
    }

    .card-stage-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .card-status-pill {
        display: inline-block;
        font-size: 0.95rem;
        font-weight: 700;
        padding: 0.35rem 0.75rem;
        border-radius: 4px;
        margin-bottom: 0.75rem;
        letter-spacing: 0.01em;
    }
    .pill-green {
        background-color: #F0FDF4;
        color: #15803D;
        border: 1px solid #BBF7D0;
    }
    .pill-red {
        background-color: #FEF2F2;
        color: #B91C1C;
        border: 1px solid #FECACA;
    }
    .pill-amber {
        background-color: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .pill-gray {
        background-color: #F9FAFB;
        color: #1F2937;
        border: 1px solid #E5E7EB;
    }

    .card-metric-text {
        font-size: 0.88rem;
        color: #111827;
        margin-bottom: 0.4rem;
    }
    .card-desc-text {
        font-size: 0.82rem;
        color: #4B5563;
        line-height: 1.45;
        margin: 0;
    }

    /* Metric Containers (Tab 3) */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-top: 3px solid #DC2626 !important;
        border-radius: 6px !important;
        padding: 1rem 1.25rem !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03) !important;
    }
    div[data-testid="stMetricLabel"] p {
        color: #4B5563 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #111827 !important;
        font-weight: 700 !important;
        font-size: 1.65rem !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        border: 1px solid #E5E7EB !important;
        border-radius: 6px !important;
        background-color: #FFFFFF !important;
        box-shadow: none !important;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 500 !important;
        color: #111827 !important;
    }

    /* Slider accent */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #DC2626 !important;
        border-color: #DC2626 !important;
    }

    /* Clean spacing */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }
</style>
""", unsafe_allow_html=True)

# Path configuration
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MAPREDUCE_DIR = os.path.join(PROJECT_DIR, 'hadoop_mapreduce')

@st.cache_resource
def load_models():
    surv_model = joblib.load(os.path.join(MODELS_DIR, 'survival_model.pkl'))
    mal_model = joblib.load(os.path.join(MODELS_DIR, 'malnutrition_model.pkl'))
    dis_model = joblib.load(os.path.join(MODELS_DIR, 'disease_model.pkl'))
    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'r') as f:
        meta = json.load(f)
    return surv_model, mal_model, dis_model, meta

try:
    surv_model, mal_model, dis_model, metadata = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please run 'python scripts/train_models.py' first.")
    st.stop()

# Header Component
st.markdown("""
<div class="app-header-container">
    <h1 class="app-title">UNICEF Child Malnutrition & Mortality Prediction System</h1>
    <p class="app-subtitle">Big Data Epidemiological Analytics on 100,000 Records using Apache Hadoop and Cascading Machine Learning Pipeline</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["Child Health Assessment", "Hadoop & MapReduce Analytics", "Model Performance & Metrics"])

# TAB 1: PREDICTION
with tabs[0]:
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="section-title">Child & Household Demographics</div>', unsafe_allow_html=True)
            age_months = st.slider("Child Age (Months)", min_value=6, max_value=59, value=18)
            gender = st.selectbox("Gender", ["Male", "Female"])
            region = st.selectbox("Region", ["Africa", "Asia", "Americas", "Europe", "Oceania"])
            residence = st.selectbox("Residence", ["Rural", "Urban"])
            household_size = st.slider("Household Size", min_value=2, max_value=10, value=5)
            household_income = st.selectbox("Household Income Level", ["Low", "Lower-Middle", "Upper-Middle", "High"])
            mother_age = st.slider("Mother Age (Years)", min_value=18, max_value=45, value=26)
            mother_edu = st.selectbox("Mother Education Level", ["Secondary", "Primary", "Higher"])

        with col2:
            st.markdown('<div class="section-title">Nutrition & Environmental Infrastructure</div>', unsafe_allow_html=True)
            birth_weight = st.number_input("Birth Weight (kg)", min_value=1.5, max_value=5.0, value=2.4, step=0.1)
            preterm_birth = st.selectbox("Preterm Birth", ["No", "Yes"])
            exclusive_bf = st.selectbox("Exclusive Breastfeeding (6 Months)", ["Yes", "No"])
            dietary_diversity = st.slider("Dietary Diversity Score (1-8 Food Groups)", min_value=1, max_value=8, value=3)
            meal_frequency = st.slider("Meal Frequency (Meals/Day)", min_value=1, max_value=6, value=2)
            food_poverty = st.selectbox("Food Poverty Household", ["Yes", "No"])
            safe_water = st.selectbox("Access to Safe Clean Water", ["No", "Yes"])
            sanitation = st.selectbox("Adequate Sanitation / Toilet", ["No", "Yes"])

        with col3:
            st.markdown('<div class="section-title">Clinical Symptoms & Health Status</div>', unsafe_allow_html=True)
            vit_a = st.selectbox("Vitamin A Supplement Received", ["No", "Yes"])
            vaccinated = st.selectbox("Vaccinated (Routine Immunization)", ["No", "Yes"])
            recent_illness = st.selectbox("Recent Illness (Past Month)", ["Yes", "No"])
            diarrhea = st.selectbox("Diarrhea in Last 2 Weeks", ["Yes", "No"])
            fever = st.selectbox("Fever in Last 2 Weeks", ["No", "Yes"])
            cough = st.selectbox("Cough in Last 2 Weeks", ["Yes", "No"])
            ari = st.selectbox("Acute Respiratory Infection (ARI) Symptoms", ["Yes", "No"])
            mother_bmi = st.number_input("Mother Body Mass Index (BMI)", min_value=15.0, max_value=35.0, value=21.0, step=0.5)

        submit_btn = st.form_submit_button("Run Cascading Health Assessment")

    if submit_btn:
        input_data = pd.DataFrame([{
            'Gender': gender,
            'Region': region,
            'Exclusive_BF_6M': exclusive_bf,
            'Safe_Water': safe_water,
            'Sanitation': sanitation,
            'Vaccinated': vaccinated,
            'Recent_Illness': recent_illness,
            'Household_Income': household_income,
            'Mother_Education': mother_edu,
            'Food_Poverty': food_poverty,
            'Residence': residence,
            'Preterm_Birth': preterm_birth,
            'Vitamin_A_Supplement': vit_a,
            'Diarrhea_Last_2Weeks': diarrhea,
            'Fever_Last_2Weeks': fever,
            'Cough_Last_2Weeks': cough,
            'ARI_Symptoms': ari,
            'Age_Months': age_months,
            'Birth_Weight_kg': birth_weight,
            'Dietary_Diversity': dietary_diversity,
            'Meal_Frequency': meal_frequency,
            'Household_Size': household_size,
            'Mother_Age': mother_age,
            'Mother_BMI': mother_bmi
        }])

        # Stage 1: Survival Prediction
        pred_surv = surv_model.predict(input_data)[0]
        prob_surv = surv_model.predict_proba(input_data)[0]

        # Stage 2: Malnutrition Prediction
        pred_mal = mal_model.predict(input_data)[0]
        prob_mal = mal_model.predict_proba(input_data)[0]

        # Stage 3: Disease Prediction
        pred_dis = dis_model.predict(input_data)[0]
        prob_dis = dis_model.predict_proba(input_data)[0]

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Cascading Diagnosis & Prognosis Results</div>', unsafe_allow_html=True)

        res_col1, res_col2, res_col3 = st.columns(3)

        # Result 1: Survival
        with res_col1:
            if pred_surv == 0:
                card_html = f"""
                <div class="clinical-card card-border-green">
                    <div>
                        <div class="card-stage-title">Stage 1: Survival Prognosis</div>
                        <div class="card-status-pill pill-green">SURVIVES (LOW RISK)</div>
                        <div class="card-metric-text"><b>Survival Probability:</b> {prob_surv[0]*100:.1f}%</div>
                    </div>
                    <p class="card-desc-text">Indicators reflect manageable baseline health conditions without immediate mortality risk.</p>
                </div>
                """
            else:
                card_html = f"""
                <div class="clinical-card card-border-red">
                    <div>
                        <div class="card-stage-title">Stage 1: Survival Prognosis</div>
                        <div class="card-status-pill pill-red">CRITICAL MORTALITY RISK</div>
                        <div class="card-metric-text"><b>Fatal Risk Probability:</b> {prob_surv[1]*100:.1f}%</div>
                    </div>
                    <p class="card-desc-text">Compounding clinical vulnerability detected (low birth weight, acute infections, un-immunized). Immediate pediatric intervention required.</p>
                </div>
                """
            st.markdown(card_html, unsafe_allow_html=True)

        # Result 2: Malnutrition Status
        with res_col2:
            if pred_mal == 0:
                m_border = "card-border-green"
                m_pill = "pill-green"
                m_title = "Normal / Well-Nourished"
                m_desc = "Nutritional intake meets standard physical development milestones. Continue routine diet counseling."
            elif pred_mal == 1:
                m_border = "card-border-amber"
                m_pill = "pill-amber"
                m_title = "Moderate Acute Malnutrition (MAM)"
                m_desc = "Targeted supplementary feeding (RUSF), micronutrient support, and active growth monitoring recommended."
            else:
                m_border = "card-border-red"
                m_pill = "pill-red"
                m_title = "Severe Acute Malnutrition (SAM)"
                m_desc = "Ready-to-Use Therapeutic Food (RUTF) protocol and structured clinical supervision required immediately."

            card_html_mal = f"""
            <div class="clinical-card {m_border}">
                <div>
                    <div class="card-stage-title">Stage 2: Malnutrition Classification</div>
                    <div class="card-status-pill {m_pill}">{m_title}</div>
                    <div class="card-metric-text"><b>Confidence Score:</b> {prob_mal[pred_mal]*100:.1f}%</div>
                </div>
                <p class="card-desc-text">{m_desc}</p>
            </div>
            """
            st.markdown(card_html_mal, unsafe_allow_html=True)

        # Result 3: Disease Prediction
        with res_col3:
            disease_map = {
                0: ("Healthy / No Acute Infection", "pill-green", "card-border-green", "No active acute infection detected. Maintain routine immunization and hygienic practices."),
                1: ("Acute Respiratory Infection (Pneumonia/ARI)", "pill-red", "card-border-red", "Assess respiratory rate and administer first-line antibiotic protocol per IMCI guidelines."),
                2: ("Diarrheal / Waterborne Illness", "pill-amber", "card-border-amber", "Administer Oral Rehydration Salts (ORS) and therapeutic zinc supplementation immediately."),
                3: ("Febrile / Systemic Illness", "pill-red", "card-border-red", "Conduct diagnostic screening for malaria or sepsis; administer antipyretic protocol as needed.")
            }
            dis_title, dis_pill, dis_border, dis_desc = disease_map.get(pred_dis, ("Unknown", "pill-gray", "card-border-gray", ""))

            card_html_dis = f"""
            <div class="clinical-card {dis_border}">
                <div>
                    <div class="card-stage-title">Stage 3: Acute Illness Diagnosis</div>
                    <div class="card-status-pill {dis_pill}">{dis_title}</div>
                    <div class="card-metric-text"><b>Diagnostic Confidence:</b> {prob_dis[pred_dis]*100:.1f}%</div>
                </div>
                <p class="card-desc-text">{dis_desc}</p>
            </div>
            """
            st.markdown(card_html_dis, unsafe_allow_html=True)

# TAB 2: HADOOP & MAPREDUCE
with tabs[1]:
    st.markdown('<div class="section-title">Apache Hadoop Distributed Architecture & MapReduce Aggregations</div>', unsafe_allow_html=True)
    st.markdown("""
    This platform integrates **Apache Hadoop** to process large-scale epidemiological surveys:
    * **Hadoop Distributed File System (HDFS):** Partitions and stores the 100,000 child survey records across distributed blocks.
    * **Hadoop Streaming MapReduce:** Executes distributed aggregation jobs using `mapper.py` and `reducer.py` to calculate demographic disease rates and malnutrition prevalence.
    """)

    report_path = os.path.join(MAPREDUCE_DIR, 'hadoop_output_report.txt')
    if os.path.exists(report_path):
        st.markdown('<div class="section-title" style="margin-top: 1.5rem;">Official Hadoop MapReduce Aggregation Report (100,000 Records)</div>', unsafe_allow_html=True)
        with open(report_path, 'r') as rf:
            rep_content = rf.read()
        st.code(rep_content, language="text")
    else:
        st.info("Execute 'python hadoop_mapreduce/run_mapreduce.py' to generate the Hadoop aggregation report.")

    with st.expander("Hadoop MapReduce Source Code (mapper.py & reducer.py)"):
        col_m, col_r = st.columns(2)
        with col_m:
            st.markdown("**Hadoop Mapper (`mapper.py`)**")
            with open(os.path.join(MAPREDUCE_DIR, 'mapper.py'), 'r') as mf:
                st.code(mf.read(), language='python')
        with col_r:
            st.markdown("**Hadoop Reducer (`reducer.py`)**")
            with open(os.path.join(MAPREDUCE_DIR, 'reducer.py'), 'r') as rf:
                st.code(rf.read(), language='python')

# TAB 3: PERFORMANCE METRICS
with tabs[2]:
    st.markdown('<div class="section-title">Model Training & Evaluation Metrics on Test Set (20,000 Children)</div>', unsafe_allow_html=True)
    
    m_col1, m_col2, m_col3 = st.columns(3)
    surv_acc = metadata['metrics']['survival']['accuracy'] * 100
    mal_acc = metadata['metrics']['malnutrition']['accuracy'] * 100
    dis_acc = metadata['metrics']['disease']['accuracy'] * 100

    m_col1.metric("Stage 1: Survival Model Accuracy", f"{surv_acc:.2f}%")
    m_col2.metric("Stage 2: Malnutrition Model Accuracy", f"{mal_acc:.2f}%")
    m_col3.metric("Stage 3: Disease Model Accuracy", f"{dis_acc:.2f}%")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    
    cm_col1, cm_col2, cm_col3 = st.columns(3)
    with cm_col1:
        st.markdown('<div class="section-title">Stage 1: Survival Matrix</div>', unsafe_allow_html=True)
        cm_surv = pd.DataFrame(
            metadata['metrics']['survival']['confusion_matrix'],
            index=['Actual: Survives', 'Actual: Critical Risk'],
            columns=['Pred: Survives', 'Pred: Critical Risk']
        )
        st.dataframe(cm_surv, use_container_width=True)

    with cm_col2:
        st.markdown('<div class="section-title">Stage 2: Malnutrition Matrix</div>', unsafe_allow_html=True)
        cm_mal = pd.DataFrame(
            metadata['metrics']['malnutrition']['confusion_matrix'],
            index=['Actual: Normal', 'Actual: MAM', 'Actual: SAM'],
            columns=['Pred: Normal', 'Pred: MAM', 'Pred: SAM']
        )
        st.dataframe(cm_mal, use_container_width=True)

    with cm_col3:
        st.markdown('<div class="section-title">Stage 3: Disease Matrix</div>', unsafe_allow_html=True)
        cm_dis = pd.DataFrame(
            metadata['metrics']['disease']['confusion_matrix'],
            index=['Actual: Healthy', 'Actual: ARI', 'Actual: Diarrheal', 'Actual: Febrile'],
            columns=['Pred: Healthy', 'Pred: ARI', 'Pred: Diarrheal', 'Pred: Febrile']
        )
        st.dataframe(cm_dis, use_container_width=True)

"""
=============================================================================
Titanic Survival Prediction System - Streamlit Web Application
=============================================================================
An interactive Machine Learning web app powered by Logistic Regression.
Features:
- Real-Time Single Passenger Survival Prediction & Risk Breakdown
- Batch CSV Prediction with Downloadable Results
- Interactive Exploratory Data Analysis (EDA) Dashboard
- Model Evaluation Metrics, ROC Curve, and Coefficient Interpretations
- Detailed Data Science Interview Q&A Section
=============================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

# --- Page Configuration ---
st.set_page_config(
    page_title="Titanic Survival Predictor | Logistic Regression",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Dark & Glassmorphic UI) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        font-size: 1.1rem;
        color: #cfd8dc;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00f2fe;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #90a4ae;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.2rem;
    }

    .survived-box {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15), rgba(39, 174, 96, 0.25));
        border: 1px solid #2ECC71;
        border-radius: 14px;
        padding: 1.5rem;
        color: #2ECC71;
        text-align: center;
    }

    .perished-box {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.15), rgba(192, 57, 43, 0.25));
        border: 1px solid #E74C3C;
        border-radius: 14px;
        padding: 1.5rem;
        color: #E74C3C;
        text-align: center;
    }

    .info-pill {
        display: inline-block;
        background: rgba(0, 242, 254, 0.1);
        color: #00f2fe;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        border: 1px solid rgba(0, 242, 254, 0.3);
        margin: 0.2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_resource
def load_model_pipeline():
    """Load the trained scikit-learn pipeline."""
    model_paths = ['titanic_logistic_pipeline.pkl', os.path.join('Logistic Regression', 'titanic_logistic_pipeline.pkl')]
    for path in model_paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None

@st.cache_data
def load_datasets():
    """Load train and test datasets."""
    train_path = 'Titanic_train.csv' if os.path.exists('Titanic_train.csv') else os.path.join('Logistic Regression', 'Titanic_train.csv')
    test_path = 'Titanic_test.csv' if os.path.exists('Titanic_test.csv') else os.path.join('Logistic Regression', 'Titanic_test.csv')
    
    df_train = pd.read_csv(train_path) if os.path.exists(train_path) else None
    df_test = pd.read_csv(test_path) if os.path.exists(test_path) else None
    return df_train, df_test

@st.cache_data
def load_metrics_and_coefficients():
    """Load precomputed metrics and feature weights."""
    metrics, coef_df = None, None
    if os.path.exists('model_metrics.json'):
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)
    if os.path.exists('model_coefficients.csv'):
        coef_df = pd.read_csv('model_coefficients.csv')
    return metrics, coef_df

def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked, title, has_cabin):
    """Format single passenger input into dataframe ready for pipeline."""
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    
    data = {
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked],
        'Title': [title],
        'FamilySize': [family_size],
        'IsAlone': [is_alone],
        'HasCabin': [has_cabin]
    }
    return pd.DataFrame(data)

def extract_batch_features(df_raw):
    """Feature engineering for uploaded batch CSV."""
    df = df_raw.copy()
    
    # 1. Title Extraction
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
            'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
            'Capt': 'Rare', 'Sir': 'Rare'
        }
        df['Title'] = df['Title'].map(title_mapping).fillna('Rare')
    elif 'Title' not in df.columns:
        df['Title'] = 'Mr'
        
    # 2. Family Size & IsAlone
    sibsp = df['SibSp'] if 'SibSp' in df.columns else 0
    parch = df['Parch'] if 'Parch' in df.columns else 0
    df['FamilySize'] = sibsp + parch + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 3. Cabin Deck Indicator
    if 'Cabin' in df.columns:
        df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
    elif 'HasCabin' not in df.columns:
        df['HasCabin'] = 0
        
    # Default columns check
    for col in ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']:
        if col not in df.columns:
            if col == 'Age': df['Age'] = 28.0
            elif col == 'Fare': df['Fare'] = 32.0
            elif col == 'Pclass': df['Pclass'] = 3
            elif col == 'Sex': df['Sex'] = 'male'
            elif col == 'Embarked': df['Embarked'] = 'S'
            elif col in ['SibSp', 'Parch']: df[col] = 0
            
    return df

# --- Load Resources ---
pipeline = load_model_pipeline()
df_train, df_test = load_datasets()
model_metrics, coef_df = load_metrics_and_coefficients()

# --- Header Section ---
st.markdown("""
<div class="main-header">
    <h1>🚢 Titanic Survival Analytics & Prediction Engine</h1>
    <p>End-to-End Binary Classification with Logistic Regression, Interpretability, and Interactive Diagnostics</p>
</div>
""", unsafe_allow_html=True)

# Top Key Performance Indicators
if model_metrics:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{model_metrics['accuracy']*100:.1f}%</div><div class="metric-label">Model Accuracy</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{model_metrics['roc_auc']:.3f}</div><div class="metric-label">ROC-AUC Score</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{model_metrics['precision']*100:.1f}%</div><div class="metric-label">Precision</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{model_metrics['recall']*100:.1f}%</div><div class="metric-label">Recall</div></div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{model_metrics['f1_score']*100:.1f}%</div><div class="metric-label">F1-Score</div></div>""", unsafe_allow_html=True)

st.write("")

# --- Navigation Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Live Passenger Predictor",
    "📁 Batch CSV Predictor",
    "📊 Exploratory Data Analysis (EDA)",
    "📈 Model Diagnostics & Coefficients",
    "💡 Technical Q&A (Interview Ready)"
])

# =============================================================================
# TAB 1: LIVE PASSENGER PREDICTOR
# =============================================================================
with tab1:
    st.subheader("Interactive Passenger Survival Simulation")
    st.markdown("Adjust the demographic, ticket, and cabin parameters below to predict the probability of passenger survival.")
    
    col_input, col_result = st.columns([3, 2], gap="large")
    
    with col_input:
        st.markdown("#### 1. Passenger Demographics")
        c1, c2, c3 = st.columns(3)
        with c1:
            title_input = st.selectbox("Honorific Title", ["Mr", "Mrs", "Miss", "Master", "Rare"], index=0,
                                       help="Extracted from Name. Master represents young boys, Mrs for married women, Miss for unmarried women.")
        with c2:
            sex_default = "female" if title_input in ["Mrs", "Miss"] else "male"
            sex_input = st.selectbox("Biological Sex", ["male", "female"], index=0 if sex_default == "male" else 1)
        with c3:
            age_input = st.slider("Age (Years)", min_value=0.5, max_value=80.0, value=29.0, step=0.5)
            
        st.markdown("#### 2. Ticket, Class & Embarkation")
        c4, c5, c6 = st.columns(3)
        with c4:
            pclass_input = st.selectbox("Ticket Class (Pclass)", [1, 2, 3], index=2,
                                        format_func=lambda x: f"{x}st Class" if x==1 else (f"{x}nd Class" if x==2 else f"{x}rd Class"))
        with c5:
            fare_input = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=600.0, value=14.5, step=1.0)
        with c6:
            embarked_input = st.selectbox("Port of Embarkation", ["S", "C", "Q"], index=0,
                                          format_func=lambda x: {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}[x])
            
        st.markdown("#### 3. Family & Accommodation")
        c7, c8, c9 = st.columns(3)
        with c7:
            sibsp_input = st.number_input("Siblings / Spouses Aboard", min_value=0, max_value=8, value=0, step=1)
        with c8:
            parch_input = st.number_input("Parents / Children Aboard", min_value=0, max_value=6, value=0, step=1)
        with c9:
            has_cabin_input = st.selectbox("Allocated Cabin?", [0, 1], index=0, format_func=lambda x: "Yes (Deck Allocated)" if x==1 else "No (NaN / Steerage)")
            
    with col_result:
        st.markdown("#### 🔮 Model Inference Result")
        
        # Prepare input sample
        input_df = preprocess_input(
            pclass=pclass_input,
            sex=sex_input,
            age=age_input,
            sibsp=sibsp_input,
            parch=parch_input,
            fare=fare_input,
            embarked=embarked_input,
            title=title_input,
            has_cabin=has_cabin_input
        )
        
        if pipeline is not None:
            prob_survive = float(pipeline.predict_proba(input_df)[0][1])
            is_survived = int(prob_survive >= 0.5)
            
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_survive * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Survival Probability", 'font': {'size': 18, 'color': '#ffffff'}},
                number={'suffix': "%", 'font': {'size': 36, 'color': '#00f2fe', 'family': 'JetBrains Mono'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#ffffff"},
                    'bar': {'color': "#00f2fe", 'thickness': 0.25},
                    'bgcolor': "rgba(255,255,255,0.05)",
                    'borderwidth': 1,
                    'bordercolor': "rgba(255,255,255,0.2)",
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(231, 76, 60, 0.4)'},
                        {'range': [40, 60], 'color': 'rgba(243, 156, 18, 0.4)'},
                        {'range': [60, 100], 'color': 'rgba(46, 204, 113, 0.4)'}
                    ],
                    'threshold': {
                        'line': {'color': "#ffffff", 'width': 3},
                        'thickness': 0.8,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(
                height=240,
                margin=dict(l=20, r=20, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "#ffffff"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Verdict Card
            if is_survived:
                st.markdown(f"""
                <div class="survived-box">
                    <h2 style="margin:0; font-size: 1.8rem;">🎉 SURVIVED</h2>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.05rem;">Estimated Survival Odds: <b>{(prob_survive/(1-prob_survive if prob_survive < 1 else 0.999)):.2f} : 1</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="perished-box">
                    <h2 style="margin:0; font-size: 1.8rem;">⚠️ PERISHED</h2>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.05rem;">Estimated Perish Probability: <b>{((1-prob_survive)*100):.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                
            # Key Factors Breakdown
            st.markdown("##### 🧬 Influential Factors for this Profile:")
            factors = []
            if sex_input == 'female' or title_input in ['Mrs', 'Miss']:
                factors.append("🟢 **Female Gender / Title**: Substantially elevated priority during evacuation.")
            if pclass_input == 1:
                factors.append("🟢 **1st Class Ticket**: Superior deck access and closer lifeboat proximity.")
            if has_cabin_input == 1:
                factors.append("🟢 **Cabin Allocation**: Strong indicator of upper deck quarters.")
            if age_input < 12 or title_input == 'Master':
                factors.append("🟢 **Child / Young Age**: 'Women and children first' maritime protocol.")
            if pclass_input == 3:
                factors.append("🔴 **3rd Class (Steerage)**: Limited egress and lower boat deck allocation.")
            if title_input == 'Mr':
                factors.append("🔴 **Adult Male ('Mr')**: Lowest evacuation priority.")
            if (sibsp_input + parch_input) >= 4:
                factors.append("🔴 **Large Family Size**: Difficulties in rallying entire party to lifeboats.")
                
            for f in factors:
                st.markdown(f)
        else:
            st.error("Model pipeline artifact not found. Please execute `python train_model.py` first.")

# =============================================================================
# TAB 2: BATCH CSV PREDICTOR
# =============================================================================
with tab2:
    st.subheader("Batch Dataset Ingestion & Mass Inference")
    st.markdown("Upload any CSV dataset containing Titanic passenger records (e.g. `Titanic_test.csv`) to generate real-time survival predictions.")
    
    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader("Upload Passenger CSV File", type=["csv"])
    with col_info:
        st.info("💡 **Required / Supported Columns**: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, `Name`, `Cabin` (Missing values are automatically imputed by the pipeline).")
        
    df_to_predict = None
    if uploaded_file is not None:
        df_to_predict = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded {len(df_to_predict)} records from uploaded file!")
    elif df_test is not None:
        if st.checkbox("Use default `Titanic_test.csv` (418 passengers)", value=True):
            df_to_predict = df_test.copy()
            st.info(f"Loaded default test split with {len(df_to_predict)} records.")
            
    if df_to_predict is not None and pipeline is not None:
        df_processed = extract_batch_features(df_to_predict)
        feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'HasCabin']
        
        preds = pipeline.predict(df_processed[feature_cols])
        probs = pipeline.predict_proba(df_processed[feature_cols])[:, 1]
        
        results_df = df_to_predict.copy()
        results_df['Predicted_Survived'] = preds
        results_df['Survival_Probability'] = np.round(probs, 4)
        results_df['Verdict'] = results_df['Predicted_Survived'].map({1: 'Survived', 0: 'Perished'})
        
        # Batch metrics summary
        surv_count = int(np.sum(preds))
        perish_count = len(preds) - surv_count
        surv_pct = (surv_count / len(preds)) * 100
        
        st.write("---")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric("Total Passengers Processed", f"{len(preds)}")
        with b2:
            st.metric("Predicted Survivors", f"{surv_count} ({surv_pct:.1f}%)")
        with b3:
            st.metric("Predicted Perished", f"{perish_count} ({(100-surv_pct):.1f}%)")
            
        # Display Results Table
        st.dataframe(
            results_df[['PassengerId', 'Name', 'Sex', 'Age', 'Pclass', 'Fare', 'Survival_Probability', 'Verdict']] if 'PassengerId' in results_df.columns else results_df,
            use_container_width=True,
            height=350
        )
        
        # CSV Download Button
        csv_bytes = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Predictions CSV",
            data=csv_bytes,
            file_name="titanic_batch_predictions.csv",
            mime="text/csv"
        )

# =============================================================================
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
with tab3:
    st.subheader("Exploratory Data Analysis & Empirical Patterns")
    st.markdown("Visualizing key demographic, socioeconomic, and familial distributions from the Titanic training set.")
    
    if df_train is not None:
        df_eda = extract_batch_features(df_train)
        
        eda_view = st.selectbox("Select Visual Exploration Focus", [
            "Demographics (Sex, Pclass & Embarkation)",
            "Continuous Distributions (Age & Fare Dynamics)",
            "Family Dynamics & Honorific Titles",
            "Multivariate Correlation Heatmap"
        ])
        
        if eda_view == "Demographics (Sex, Pclass & Embarkation)":
            col_a, col_b = st.columns(2)
            with col_a:
                fig_sex = px.histogram(
                    df_eda, x="Sex", color="Survived", barmode="group",
                    color_discrete_map={0: "#E74C3C", 1: "#2ECC71"},
                    labels={"Survived": "Survival Status"},
                    title="<b>Survival Distribution by Biological Sex</b>"
                )
                fig_sex.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sex, use_container_width=True)
                
            with col_b:
                fig_pclass = px.histogram(
                    df_eda, x="Pclass", color="Survived", barmode="group",
                    color_discrete_map={0: "#E74C3C", 1: "#2ECC71"},
                    labels={"Pclass": "Passenger Class (1=1st, 2=2nd, 3=3rd)", "Survived": "Survival Status"},
                    title="<b>Survival Distribution by Ticket Class</b>"
                )
                fig_pclass.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_pclass, use_container_width=True)
                
            fig_emb = px.histogram(
                df_eda, x="Embarked", color="Survived", barmode="group",
                color_discrete_map={0: "#E74C3C", 1: "#2ECC71"},
                labels={"Embarked": "Port of Embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)"},
                title="<b>Survival Count across Embarkation Ports</b>"
            )
            fig_emb.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_emb, use_container_width=True)
            
        elif eda_view == "Continuous Distributions (Age & Fare Dynamics)":
            col_c, col_d = st.columns(2)
            with col_c:
                fig_age = px.histogram(
                    df_eda, x="Age", color="Survived", marginal="box",
                    color_discrete_map={0: "#E74C3C", 1: "#2ECC71"},
                    nbins=35,
                    title="<b>Age Distribution vs Survival Outcome</b>"
                )
                fig_age.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_age, use_container_width=True)
                
            with col_d:
                fig_fare = px.histogram(
                    df_eda, x="Fare", color="Survived", marginal="box",
                    color_discrete_map={0: "#E74C3C", 1: "#2ECC71"},
                    nbins=40,
                    title="<b>Fare Distribution vs Survival Outcome</b>"
                )
                fig_fare.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_fare, use_container_width=True)
                
            # Class vs Fare vs Survival Scatter
            fig_scatter = px.scatter(
                df_eda, x="Age", y="Fare", color="Survived", size="FamilySize",
                color_discrete_map={0: "#E74C3C", 1: "#2ECC71"},
                hover_data=["Name", "Pclass", "Title"],
                title="<b>Multidimensional View: Age vs Fare by Survival & Family Size</b>"
            )
            fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        elif eda_view == "Family Dynamics & Honorific Titles":
            col_e, col_f = st.columns(2)
            with col_e:
                fam_df = df_eda.groupby('FamilySize')['Survived'].mean().reset_index()
                fig_fam = px.bar(
                    fam_df, x="FamilySize", y="Survived",
                    color="Survived", color_continuous_scale="Viridis",
                    labels={"Survived": "Survival Rate", "FamilySize": "Total Family Size (SibSp + Parch + 1)"},
                    title="<b>Survival Rate by Family Size (Peak at 2 to 4 Members)</b>"
                )
                fig_fam.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_fam, use_container_width=True)
                
            with col_f:
                title_df = df_eda.groupby('Title')['Survived'].mean().reset_index().sort_values(by='Survived', ascending=False)
                fig_title = px.bar(
                    title_df, x="Title", y="Survived",
                    color="Survived", color_continuous_scale="Blues",
                    labels={"Survived": "Survival Rate"},
                    title="<b>Survival Rate by Extracted Title</b>"
                )
                fig_title.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_title, use_container_width=True)
                
        elif eda_view == "Multivariate Correlation Heatmap":
            num_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'HasCabin']
            corr = df_eda[num_cols].corr()
            fig_corr = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu_r",
                title="<b>Pearson Correlation Heatmap of Numeric & Extracted Features</b>"
            )
            fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_corr, use_container_width=True)

# =============================================================================
# TAB 4: MODEL DIAGNOSTICS & COEFFICIENTS
# =============================================================================
with tab4:
    st.subheader("Model Diagnostic Curves & Mathematical Interpretation")
    st.markdown("Detailed breakdown of logistic regression parameter weights, odds ratios, and classification diagnostics.")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        if os.path.exists('plots/model_roc_curve.png'):
            st.image('plots/model_roc_curve.png', caption="ROC Curve (Validation AUC = 0.868)", use_container_width=True)
        elif model_metrics:
            st.info("ROC Curve generated from validation data.")
            
    with col_m2:
        if os.path.exists('plots/model_confusion_matrix.png'):
            st.image('plots/model_confusion_matrix.png', caption="Validation Confusion Matrix", use_container_width=True)
            
    st.write("---")
    st.markdown("### ⚖️ Coefficient & Odds Ratio ($e^{\\beta}$) Breakdown")
    
    if coef_df is not None:
        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            fig_coef = px.bar(
                coef_df.sort_values(by='Coefficient'),
                x='Coefficient', y='Feature', orientation='h',
                color='Coefficient',
                color_continuous_scale=['#E74C3C', '#2ECC71'],
                title="<b>Model Feature Weights (Log-Odds Impact $\\beta$)</b>"
            )
            fig_coef.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_coef, use_container_width=True)
            
        with col_c2:
            st.markdown("#### Tabular Weights & Odds Ratios")
            st.dataframe(
                coef_df[['Feature', 'Coefficient', 'Odds_Ratio']].style.format({
                    'Coefficient': '{:.4f}',
                    'Odds_Ratio': '{:.4f}'
                }),
                use_container_width=True,
                height=380
            )
            
        st.markdown("""
        #### 📖 Mathematical Interpretation:
        1. **Log-Odds Equation**:
           $$\\ln\\left(\\frac{P(Y=1)}{1 - P(Y=1)}\\right) = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\dots + \\beta_k X_k$$
        2. **Odds Ratio ($e^{\\beta}$)**:
           - An Odds Ratio **$> 1$** indicates that as the feature increases (or is present), the odds of survival increase. For example, `HasCabin` has an Odds Ratio of **3.07**, meaning passengers with known cabins had **~3.07 times the odds** of surviving compared to those without.
           - An Odds Ratio **$< 1$** indicates a reduction in survival odds. For instance, `Title_Mr` ($\beta = -2.046$, Odds Ratio $= 0.129$) indicates adult males faced an **87.1% reduction in odds of survival** compared to baseline.
           - `Pclass_3` ($\beta = -1.081$, Odds Ratio $= 0.339$) shows 3rd class passengers had **66.1% lower survival odds** relative to 1st class.
        """)

# =============================================================================
# TAB 5: TECHNICAL Q&A / INTERVIEW QUESTIONS
# =============================================================================
with tab5:
    st.subheader("🎓 Technical Data Science Interview Questions")
    st.markdown("Comprehensive, interview-ready answers with mathematical formulas and practical classification insights.")
    
    with st.expander("❓ Question 1: What is the difference between Precision and Recall?", expanded=True):
        st.markdown("""
        ### Precision vs. Recall: Definitions, Trade-offs, and Context
        
        #### 1. Mathematical Definitions:
        - **Precision (Positive Predictive Value)**:
          $$\\text{Precision} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP}}$$
          *Answers*: **Of all instances predicted as Positive, how many are actually Positive?**
          
        - **Recall (Sensitivity / True Positive Rate)**:
          $$\\text{Recall} = \\frac{\\text{TP}}{\\text{TP} + \\text{FN}}$$
          *Answers*: **Of all actual Positive instances in reality, how many did the model capture?**
          
        #### 2. The Precision-Recall Trade-off:
        - By adjusting the classification decision threshold ($\tau$):
          - **Increasing $\\tau$ (e.g., from 0.5 to 0.8)** $\\rightarrow$ The model becomes more conservative. False Positives decrease, so **Precision increases**, but False Negatives rise, causing **Recall to decrease**.
          - **Decreasing $\\tau$ (e.g., from 0.5 to 0.2)** $\\rightarrow$ The model becomes more aggressive. True Positives increase and False Negatives decrease (**Recall increases**), but False Positives rise, causing **Precision to drop**.
          
        #### 3. Real-World Business Contexts:
        | Scenario | Priority Metric | Rationale |
        | :--- | :--- | :--- |
        | **Cancer / Disease Diagnosis** | **Recall** | A False Negative (missing a sick patient) is catastrophic; a False Positive merely prompts a secondary test. |
        | **Spam Detection / Fraud Flagging** | **Precision** | A False Positive (sending an important email to spam) directly disrupts the user experience. |
        | **Titanic Evacuation Priority** | **Balanced ($F_1$-score)** | Balancing scarce lifeboat seats (avoiding FP waste) with saving lives (avoiding FN abandonment). |
        
        #### 4. Harmonic Mean ($F_1$-Score):
        $$F_1 = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$
        The harmonic mean severely penalizes extreme imbalances between Precision and Recall.
        """)
        
    with st.expander("❓ Question 2: What is Cross-Validation, and why is it important in binary classification?", expanded=True):
        st.markdown("""
        ### Cross-Validation in Binary Classification
        
        #### 1. What is Cross-Validation?
        Cross-Validation (CV) is a statistical resampling technique used to evaluate model generalization and mitigate overfitting. The dataset is partitioned into $K$ non-overlapping folds; iteratively, $K-1$ folds are used for training and the remaining 1 fold serves as the validation set. The final performance estimate is the average across all $K$ iterations.
        
        $$\\text{CV Score} = \\frac{1}{K} \\sum_{k=1}^K \\text{Metric}_k$$
        
        #### 2. Why Stratified $K$-Fold is Essential for Binary Classification:
        - **Preserving Class Distribution**: In binary classification, target classes are frequently imbalanced (e.g., 38% survived vs 62% perished in Titanic, or 1% fraud vs 99% non-fraud). Regular $K$-Fold might randomly produce a validation fold with few or zero positive cases.
        - **Stratified $K$-Fold** enforces that every single fold preserves the exact class ratio of the full dataset:
          $$\\left(\\frac{\\text{Class 1}}{\\text{Class 0}}\\right)_{\\text{Fold}_k} = \\left(\\frac{\\text{Class 1}}{\\text{Class 0}}\\right)_{\\text{Dataset}}$$
          
        #### 3. Key Benefits in Production Machine Learning:
        1. **Prevents Optimistic Bias / Data Leakage**: Evaluates models purely on unseen validation slices.
        2. **Quantifies Variance & Model Stability**: By examining the standard deviation of scores across folds ($\\sigma_{\\text{CV}}$), engineers can verify whether the model is sensitive to training set perturbations.
        3. **Reliable Hyperparameter Tuning**: During `GridSearchCV`, tuning on CV scores ensures parameters generalize well without overfitting the validation split.
        """)

# --- Footer ---
st.write("---")
st.markdown("""
<div style="text-align: center; color: #78909c; font-size: 0.85rem;">
    🚢 Titanic Logistic Regression Analytics Engine &bull; Built with Streamlit, Scikit-Learn & Plotly &bull; End-to-End Machine Learning
</div>
""", unsafe_allow_html=True)

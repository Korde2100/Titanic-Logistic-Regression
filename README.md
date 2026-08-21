# 🚢 Titanic Survival Prediction using Logistic Regression

An end-to-end Data Science and Machine Learning project implementing **Logistic Regression** for binary classification on the Titanic dataset, paired with an interactive **Streamlit** web application and comprehensive model diagnostics.

---

## 📌 Project Overview
This project delivers a complete machine learning lifecycle solution:
1. **Data Exploration & Visualizations (EDA)**: Statistical distributions, missingness analysis, demographic breakdowns, and correlation heatmaps.
2. **Data Preprocessing & Feature Engineering**: Missing value imputation, honorific title extraction, family size dynamics (`FamilySize`, `IsAlone`), cabin detection (`HasCabin`), and standard scaling within a leakage-free Scikit-Learn `ColumnTransformer`.
3. **Model Building & Cross-Validation**: Logistic Regression tuned via 5-Fold `StratifiedKFold` cross-validation with `GridSearchCV`.
4. **Model Evaluation & Visualizations**: In-depth metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC) accompanied by publication-quality ROC curves, Confusion Matrices, and Precision-Recall curves.
5. **Model Interpretation**: Mathematical log-odds parameter interpretation and Odds Ratios ($e^{\beta}$) analysis.
6. **Deployment with Streamlit**: Interactive web dashboard featuring real-time single passenger predictions, batch CSV inference, exploratory charts, and live model diagnostics.
7. **Interview Q&A**: In-depth answers to core binary classification technical interview questions.

---
  **LIVE DEMO :** ```https://titanic-logistic-regression-94bctmxdfaoypxokoqkuv7.streamlit.app/``` 
## 📂 Repository Structure
```
├── Titanic_train.csv                   # Raw training dataset (891 records)
├── Titanic_test.csv                    # Raw testing dataset (418 records)
├── train_model.py                      # Master pipeline training & evaluation script
├── app.py                              # Interactive Streamlit Web Application
├── logistic_regression_titanic.ipynb   # Step-by-step Jupyter Notebook
├── titanic_logistic_pipeline.pkl       # Serialized Scikit-Learn model pipeline
├── titanic_test_predictions.csv        # Generated predictions on Titanic_test.csv
├── model_coefficients.csv              # Model weights and odds ratios
├── model_metrics.json                  # Serialized evaluation metrics
├── requirements.txt                    # Project dependencies
├── plots/                              # Generated high-resolution diagnostic plots
│   ├── eda_survival_demographics.png
│   ├── eda_age_fare_distribution.png
│   ├── eda_family_effects.png
│   ├── eda_correlation_heatmap.png
│   ├── model_confusion_matrix.png
│   ├── model_roc_curve.png
│   ├── model_precision_recall_curve.png
│   └── model_coefficients_odds_ratios.png
└── README.md                           # Documentation & deployment guide
```

---

## 📊 Model Performance Summary

| Metric | Cross-Validation (5-Fold Mean) | Validation Set (Hold-out 20%) |
| :--- | :--- | :--- |
| **Accuracy** | **81.2%** ($\pm 1.8\%$) | **81.6%** |
| **ROC-AUC Score** | **0.871** | **0.868** |
| **Precision** | **78.4%** | **79.0%** |
| **Recall (Sensitivity)** | **71.3%** | **71.0%** |
| **$F_1$-Score** | **0.745** | **0.748** |

### Top Features by Odds Ratio ($e^\beta$)
- **`HasCabin_1`** ($\beta = +1.123$, $\text{OR} = 3.07$): Having an assigned cabin increased survival odds by over **300%**.
- **`Title_Mrs`** ($\beta = +0.496$, $\text{OR} = 1.64$): Married women had **64% higher odds** of surviving.
- **`Pclass_3`** ($\beta = -1.081$, $\text{OR} = 0.34$): 3rd class passengers experienced a **66% decrease in survival odds** relative to 1st class.
- **`Title_Mr`** ($\beta = -2.046$, $\text{OR} = 0.13$): Adult male passengers faced an **87% reduction in odds** of survival, directly reflecting the evacuation priority given to women and children.

---

## 🚀 Getting Started

### 1. Installation & Environment Setup
Clone or navigate to the project directory, then install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Retraining the Pipeline & Generating Visualizations
To run the end-to-end data pipeline, generate plots, evaluate metrics, and save the serialized model:
```bash
python train_model.py
```

### 3. Running the Jupyter Notebook
Open and execute `logistic_regression_titanic.ipynb` in your preferred notebook interface:
```bash
jupyter notebook logistic_regression_titanic.ipynb
```

### 4. Launching the Streamlit Web Application Locally
Start the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🌐 Online Deployment Guide (Streamlit Community Cloud)

To deploy this app publicly on **Streamlit Community Cloud** ([docs.streamlit.io](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)):
1. Push this repository to a **GitHub** repository.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
3. Click **"New App"** and select:
   - **Repository**: Your GitHub repository name
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **Deploy!** — Streamlit Cloud will install dependencies from `requirements.txt` and launch your live application with a public URL.

---

## 💡 Data Science Interview Answers

### Question 1: What is the difference between Precision and Recall?
- **Precision (Positive Predictive Value)**:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
  *Answers*: Of all passengers predicted to survive, how many actually survived? Focuses on minimizing False Positives.
  
- **Recall (Sensitivity / True Positive Rate)**:
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  *Answers*: Of all passengers who actually survived, how many did the model correctly identify? Focuses on minimizing False Negatives.

- **Trade-Off**: Adjusting the classification decision threshold shifts this trade-off. Increasing the threshold increases precision at the cost of lower recall. The **$F_1$-Score** provides a harmonic mean balance when both types of errors are costly.

### Question 2: What is Cross-Validation, and why is it important in binary classification?
- **Definition**: Resampling method where the dataset is split into $K$ folds. The model is trained on $K-1$ folds and validated on the remaining fold, rotating across all $K$ iterations to produce an averaged score $\frac{1}{K}\sum \text{Score}_k$.
- **Importance in Binary Classification**:
  1. **Stratified $K$-Fold**: Guarantees that the proportion of positive and negative classes is identical in every fold, preventing unrepresentative validation splits.
  2. **Prevents Overfitting & Data Leakage**: All transformations (imputation, scaling, encoding) and hyperparameter search (`GridSearchCV`) occur strictly within training folds.
  3. **Variance Estimation**: Standard deviation across folds quantifies how sensitive model performance is to training data variations.

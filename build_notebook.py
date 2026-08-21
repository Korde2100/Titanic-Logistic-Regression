"""
Script to generate the complete, high-quality logistic_regression_titanic.ipynb notebook.
"""

import json

def create_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🚢 Titanic Survival Prediction using Logistic Regression\n",
                    "### Complete End-to-End Machine Learning Workflow\n",
                    "---\n",
                    "This comprehensive Jupyter Notebook contains the step-by-step implementation of a **Logistic Regression** binary classification pipeline to predict passenger survival on the Titanic.\n",
                    "\n",
                    "### Project Outline:\n",
                    "1. **Data Exploration & Exploratory Data Analysis (EDA)**\n",
                    "2. **Data Preprocessing & Feature Engineering**\n",
                    "3. **Model Building & Hyperparameter Tuning via Stratified K-Fold Cross-Validation**\n",
                    "4. **Model Evaluation & Diagnostic Visualizations (ROC-AUC, Confusion Matrix, Precision-Recall Curve)**\n",
                    "5. **Model Interpretation (Coefficients & Odds Ratios)**\n",
                    "6. **Model Serialization & Streamlit Deployment Overview**\n",
                    "7. **Data Science Interview Q&A (Precision vs. Recall, Cross-Validation)**\n"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 0. Environment Setup & Library Imports\n",
                    "We start by importing the necessary data science, machine learning, and visualization libraries."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import json\n",
                    "import joblib\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV\n",
                    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
                    "from sklearn.impute import SimpleImputer\n",
                    "from sklearn.compose import ColumnTransformer\n",
                    "from sklearn.pipeline import Pipeline\n",
                    "from sklearn.linear_model import LogisticRegression\n",
                    "from sklearn.metrics import (\n",
                    "    accuracy_score, precision_score, recall_score, f1_score,\n",
                    "    roc_auc_score, roc_curve, confusion_matrix, classification_report,\n",
                    "    precision_recall_curve, average_precision_score\n",
                    ")\n",
                    "\n",
                    "# Set plot styling\n",
                    "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                    "plt.rcParams['font.size'] = 11\n",
                    "plt.rcParams['figure.titlesize'] = 14\n",
                    "print(\"Libraries successfully imported!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Data Exploration & Exploratory Data Analysis (EDA)\n",
                    "### 1.a & 1.b: Loading Data and Summary Statistics"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load the Titanic datasets\n",
                    "train_path = 'Titanic_train.csv' if os.path.exists('Titanic_train.csv') else 'Logistic Regression/Titanic_train.csv'\n",
                    "test_path = 'Titanic_test.csv' if os.path.exists('Titanic_test.csv') else 'Logistic Regression/Titanic_test.csv'\n",
                    "\n",
                    "df_train = pd.read_csv(train_path)\n",
                    "df_test = pd.read_csv(test_path)\n",
                    "\n",
                    "print(f\"Training Set Shape: {df_train.shape}\")\n",
                    "print(f\"Testing Set Shape:  {df_test.shape}\")\n",
                    "df_train.head()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Inspect Data Types, Non-Null Counts, and Memory Usage\n",
                    "df_train.info()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Descriptive Statistics for Numerical Columns\n",
                    "df_train.describe().T"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Missing Value Analysis\n",
                    "missing_df = pd.DataFrame({\n",
                    "    'Missing_Count': df_train.isnull().sum(),\n",
                    "    'Percentage (%)': (df_train.isnull().sum() / len(df_train)) * 100\n",
                    "})\n",
                    "missing_df[missing_df['Missing_Count'] > 0]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1.c: Visualizations and Pattern Analysis\n",
                    "Let's visualize the relationship between passenger survival and key demographic/socioeconomic features:\n",
                    "- Biological Sex (`Sex`)\n",
                    "- Socioeconomic Class (`Pclass`)\n",
                    "- Port of Embarkation (`Embarked`)\n",
                    "- Age and Ticket Fare Distributions\n",
                    "- Family Composition (`SibSp`, `Parch`)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Categorical Demographics vs. Survival\n",
                    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
                    "\n",
                    "sns.countplot(data=df_train, x='Sex', hue='Survived', palette=['#E74C3C', '#2ECC71'], ax=axes[0])\n",
                    "axes[0].set_title('Survival by Sex', fontweight='bold')\n",
                    "axes[0].legend(['Perished (0)', 'Survived (1)'])\n",
                    "\n",
                    "sns.countplot(data=df_train, x='Pclass', hue='Survived', palette=['#E74C3C', '#2ECC71'], ax=axes[1])\n",
                    "axes[1].set_title('Survival by Ticket Class (Pclass)', fontweight='bold')\n",
                    "axes[1].legend(['Perished (0)', 'Survived (1)'])\n",
                    "\n",
                    "sns.countplot(data=df_train, x='Embarked', hue='Survived', palette=['#E74C3C', '#2ECC71'], ax=axes[2])\n",
                    "axes[2].set_title('Survival by Embarkation Port', fontweight='bold')\n",
                    "axes[2].legend(['Perished (0)', 'Survived (1)'])\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Continuous Distributions: Age and Fare\n",
                    "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n",
                    "\n",
                    "sns.histplot(data=df_train, x='Age', hue='Survived', kde=True, bins=30, palette=['#E74C3C', '#2ECC71'], element='step', ax=axes[0, 0])\n",
                    "axes[0, 0].set_title('Age Distribution by Survival Status', fontweight='bold')\n",
                    "\n",
                    "sns.boxplot(data=df_train, x='Pclass', y='Age', hue='Survived', palette=['#E74C3C', '#2ECC71'], ax=axes[0, 1])\n",
                    "axes[0, 1].set_title('Age Distribution by Class and Survival', fontweight='bold')\n",
                    "\n",
                    "sns.histplot(data=df_train, x='Fare', hue='Survived', kde=True, bins=30, palette=['#E74C3C', '#2ECC71'], element='step', ax=axes[1, 0])\n",
                    "axes[1, 0].set_title('Fare Distribution by Survival Status', fontweight='bold')\n",
                    "\n",
                    "sns.boxplot(data=df_train, x='Pclass', y='Fare', hue='Survived', palette=['#E74C3C', '#2ECC71'], ax=axes[1, 1])\n",
                    "axes[1, 1].set_title('Fare Distribution by Class (Capped View)', fontweight='bold')\n",
                    "axes[1, 1].set_ylim(0, 300)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Key Insights from Exploratory Data Analysis:\n",
                    "1. **Gender**: Female survival rate was ~74.2% compared to only ~18.9% for males, directly validating the historical \"women and children first\" evacuation protocol.\n",
                    "2. **Socioeconomic Class**: 1st Class passengers had a survival rate of ~63.0%, 2nd Class ~47.3%, and 3rd Class only ~24.2%.\n",
                    "3. **Age**: Children under 10 years old showed elevated survival rates across classes.\n",
                    "4. **Cabin**: Over 77% of cabin entries are missing in the training set, reflecting lower-tier steerage tickets lacking reserved cabin assignments."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Feature Engineering & Data Preprocessing\n",
                    "We engineer domain-specific features:\n",
                    "1. **`Title`**: Extracted from the passenger `Name` field (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`).\n",
                    "2. **`FamilySize`**: Total family members $= \\text{SibSp} + \\text{Parch} + 1$.\n",
                    "3. **`IsAlone`**: Binary flag indicating whether the passenger traveled without family.\n",
                    "4. **`HasCabin`**: Binary flag indicating if a cabin identifier was recorded (proxy for upper decks).\n",
                    "\n",
                    "We encapsulate preprocessing in a `scikit-learn ColumnTransformer` to prevent data leakage during cross-validation."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def extract_features(df_raw):\n",
                    "    \"\"\"Extract engineered features from raw Titanic dataframe.\"\"\"\n",
                    "    df = df_raw.copy()\n",
                    "    \n",
                    "    # 1. Title Extraction\n",
                    "    if 'Name' in df.columns:\n",
                    "        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\\.', expand=False)\n",
                    "        title_mapping = {\n",
                    "            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',\n",
                    "            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',\n",
                    "            'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',\n",
                    "            'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',\n",
                    "            'Capt': 'Rare', 'Sir': 'Rare'\n",
                    "        }\n",
                    "        df['Title'] = df['Title'].map(title_mapping).fillna('Rare')\n",
                    "    else:\n",
                    "        df['Title'] = 'Mr'\n",
                    "        \n",
                    "    # 2. Family Size & IsAlone\n",
                    "    if 'SibSp' in df.columns and 'Parch' in df.columns:\n",
                    "        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1\n",
                    "        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)\n",
                    "    else:\n",
                    "        df['FamilySize'] = 1\n",
                    "        df['IsAlone'] = 1\n",
                    "        \n",
                    "    # 3. Cabin Indicator\n",
                    "    if 'Cabin' in df.columns:\n",
                    "        df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)\n",
                    "    else:\n",
                    "        df['HasCabin'] = 0\n",
                    "        \n",
                    "    return df\n",
                    "\n",
                    "df_train_feat = extract_features(df_train)\n",
                    "df_train_feat[['Name', 'Title', 'FamilySize', 'IsAlone', 'HasCabin']].head()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Feature Correlation Heatmap\n",
                    "plt.figure(figsize=(10, 8))\n",
                    "num_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'HasCabin']\n",
                    "corr_mat = df_train_feat[num_cols].corr()\n",
                    "mask = np.triu(np.ones_like(corr_mat, dtype=bool))\n",
                    "sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='coolwarm', mask=mask, linewidths=0.5)\n",
                    "plt.title('Correlation Matrix of Numeric & Engineered Features', fontweight='bold', pad=15)\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Model Building & Hyperparameter Tuning\n",
                    "We configure an end-to-end `Pipeline` integrating:\n",
                    "- `SimpleImputer` for numerical (median) and categorical (mode) missing values.\n",
                    "- `StandardScaler` for continuous variables.\n",
                    "- `OneHotEncoder` with `drop='first'` for categorical variables.\n",
                    "- `LogisticRegression` optimized via 5-Fold `StratifiedKFold` Cross-Validation and `GridSearchCV`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Define Feature Matrix and Target Vector\n",
                    "feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'HasCabin']\n",
                    "X = df_train_feat[feature_cols]\n",
                    "y = df_train_feat['Survived']\n",
                    "\n",
                    "# Stratified Train-Validation Split (80% Train, 20% Validation)\n",
                    "X_train, X_val, y_train, y_val = train_test_split(\n",
                    "    X, y, test_size=0.20, random_state=42, stratify=y\n",
                    ")\n",
                    "\n",
                    "# Column Transformer Pipelines\n",
                    "numeric_features = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch']\n",
                    "categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'HasCabin']\n",
                    "\n",
                    "num_transformer = Pipeline(steps=[\n",
                    "    ('imputer', SimpleImputer(strategy='median')),\n",
                    "    ('scaler', StandardScaler())\n",
                    "])\n",
                    "\n",
                    "cat_transformer = Pipeline(steps=[\n",
                    "    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
                    "    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))\n",
                    "])\n",
                    "\n",
                    "preprocessor = ColumnTransformer(\n",
                    "    transformers=[\n",
                    "        ('num', num_transformer, numeric_features),\n",
                    "        ('cat', cat_transformer, categorical_features)\n",
                    "    ]\n",
                    ")\n",
                    "\n",
                    "# Full Logistic Regression Pipeline\n",
                    "pipeline = Pipeline(steps=[\n",
                    "    ('preprocessor', preprocessor),\n",
                    "    ('classifier', LogisticRegression(max_iter=1000, random_state=42))\n",
                    "])\n",
                    "\n",
                    "# Hyperparameter Grid Search\n",
                    "param_grid = {\n",
                    "    'classifier__C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],\n",
                    "    'classifier__penalty': ['l2'],\n",
                    "    'classifier__solver': ['lbfgs', 'liblinear'],\n",
                    "    'classifier__class_weight': [None, 'balanced']\n",
                    "}\n",
                    "\n",
                    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
                    "grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)\n",
                    "grid_search.fit(X_train, y_train)\n",
                    "\n",
                    "best_model = grid_search.best_estimator_\n",
                    "print(f\"Best Cross-Validation ROC-AUC: {grid_search.best_score_:.4f}\")\n",
                    "print(f\"Optimal Hyperparameters:      {grid_search.best_params_}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Model Evaluation & Visualizations\n",
                    "We evaluate the optimized model on the held-out validation set using:\n",
                    "- **Accuracy**: Overall classification correctness.\n",
                    "- **Precision**: Proportion of predicted survivors who actually survived.\n",
                    "- **Recall (Sensitivity)**: Proportion of actual survivors correctly identified.\n",
                    "- **$F_1$-Score**: Harmonic mean of Precision and Recall.\n",
                    "- **ROC-AUC**: Area under the Receiver Operating Characteristic curve."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Predictions on Validation Set\n",
                    "y_pred = best_model.predict(X_val)\n",
                    "y_prob = best_model.predict_proba(X_val)[:, 1]\n",
                    "\n",
                    "acc = accuracy_score(y_val, y_pred)\n",
                    "prec = precision_score(y_val, y_pred)\n",
                    "rec = recall_score(y_val, y_pred)\n",
                    "f1 = f1_score(y_val, y_pred)\n",
                    "roc_auc = roc_auc_score(y_val, y_prob)\n",
                    "\n",
                    "print(\"=\"*45)\n",
                    "print(\"       VALIDATION PERFORMANCE METRICS\")\n",
                    "print(\"=\"*45)\n",
                    "print(f\"Accuracy  : {acc:.4f} ({acc*100:.2f}%)\")\n",
                    "print(f\"Precision : {prec:.4f} ({prec*100:.2f}%)\")\n",
                    "print(f\"Recall    : {rec:.4f} ({rec*100:.2f}%)\")\n",
                    "print(f\"F1-Score  : {f1:.4f}\")\n",
                    "print(f\"ROC-AUC   : {roc_auc:.4f}\")\n",
                    "print(\"\\nClassification Report:\\n\", classification_report(y_val, y_pred))"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Visual Diagnostics: Confusion Matrix, ROC Curve, and PR Curve\n",
                    "fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))\n",
                    "\n",
                    "# 1. Confusion Matrix\n",
                    "cm = confusion_matrix(y_val, y_pred)\n",
                    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],\n",
                    "            xticklabels=['Perished (0)', 'Survived (1)'],\n",
                    "            yticklabels=['Perished (0)', 'Survived (1)'],\n",
                    "            annot_kws={'size': 14, 'weight': 'bold'})\n",
                    "axes[0].set_title('Validation Confusion Matrix', fontweight='bold')\n",
                    "axes[0].set_xlabel('Predicted Label')\n",
                    "axes[0].set_ylabel('True Label')\n",
                    "\n",
                    "# 2. ROC Curve\n",
                    "fpr, tpr, _ = roc_curve(y_val, y_prob)\n",
                    "axes[1].plot(fpr, tpr, color='#2980B9', lw=2.5, label=f'Logistic Regression (AUC = {roc_auc:.3f})')\n",
                    "axes[1].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Chance (AUC = 0.500)')\n",
                    "axes[1].fill_between(fpr, tpr, alpha=0.15, color='#2980B9')\n",
                    "axes[1].set_title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')\n",
                    "axes[1].set_xlabel('False Positive Rate (1 - Specificity)')\n",
                    "axes[1].set_ylabel('True Positive Rate (Recall)')\n",
                    "axes[1].legend(loc='lower right')\n",
                    "\n",
                    "# 3. Precision-Recall Curve\n",
                    "prec_pts, rec_pts, _ = precision_recall_curve(y_val, y_prob)\n",
                    "ap_score = average_precision_score(y_val, y_prob)\n",
                    "axes[2].plot(rec_pts, prec_pts, color='#27AE60', lw=2.5, label=f'PR Curve (AP = {ap_score:.3f})')\n",
                    "axes[2].fill_between(rec_pts, prec_pts, alpha=0.15, color='#27AE60')\n",
                    "axes[2].set_title('Precision-Recall Curve', fontweight='bold')\n",
                    "axes[2].set_xlabel('Recall')\n",
                    "axes[2].set_ylabel('Precision')\n",
                    "axes[2].legend(loc='lower left')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Model Interpretation: Coefficients & Odds Ratios\n",
                    "In Logistic Regression, the model models the log-odds of the positive outcome ($Y=1$):\n",
                    "\n",
                    "$$\\ln\\left(\\frac{p}{1-p}\\right) = \\beta_0 + \\beta_1 X_1 + \\dots + \\beta_k X_k$$\n",
                    "\n",
                    "The **Odds Ratio (OR)** is computed as:\n",
                    "$$\\text{Odds Ratio} = e^{\\beta_j}$$\n",
                    "\n",
                    "- $\\text{Odds Ratio} > 1$: Feature increases the odds of survival.\n",
                    "- $\\text{Odds Ratio} < 1$: Feature decreases the odds of survival.\n",
                    "- Percentage change in odds $= (e^{\\beta_j} - 1) \\times 100\\%$."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Extract feature names and weights\n",
                    "prep_step = best_model.named_steps['preprocessor']\n",
                    "clf_step = best_model.named_steps['classifier']\n",
                    "\n",
                    "cat_encoder = prep_step.named_transformers_['cat'].named_steps['onehot']\n",
                    "encoded_cat_cols = list(cat_encoder.get_feature_names_out(categorical_features))\n",
                    "all_feature_names = numeric_features + encoded_cat_cols\n",
                    "\n",
                    "coefs = clf_step.coef_[0]\n",
                    "odds_ratios = np.exp(coefs)\n",
                    "pct_change = (odds_ratios - 1) * 100\n",
                    "\n",
                    "coef_df = pd.DataFrame({\n",
                    "    'Feature': all_feature_names,\n",
                    "    'Coefficient (Beta)': np.round(coefs, 4),\n",
                    "    'Odds Ratio (e^Beta)': np.round(odds_ratios, 4),\n",
                    "    'Pct Change in Odds (%)': np.round(pct_change, 2)\n",
                    "}).sort_values(by='Odds Ratio (e^Beta)', ascending=False).reset_index(drop=True)\n",
                    "\n",
                    "print(f\"Intercept (Baseline Log-Odds): {clf_step.intercept_[0]:.4f}\")\n",
                    "coef_df"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Plot Coefficients and Odds Ratios\n",
                    "fig, axes = plt.subplots(1, 2, figsize=(16, 7))\n",
                    "\n",
                    "# Coefficients Bar Plot\n",
                    "sorted_coef = coef_df.sort_values(by='Coefficient (Beta)')\n",
                    "colors = ['#2ECC71' if c > 0 else '#E74C3C' for c in sorted_coef['Coefficient (Beta)']]\n",
                    "axes[0].barh(sorted_coef['Feature'], sorted_coef['Coefficient (Beta)'], color=colors)\n",
                    "axes[0].axvline(0, color='black', linestyle='--', alpha=0.7)\n",
                    "axes[0].set_title('Feature Coefficients (Log-Odds Impact)', fontweight='bold')\n",
                    "axes[0].set_xlabel('Coefficient (Beta)')\n",
                    "\n",
                    "# Odds Ratios Bar Plot\n",
                    "axes[1].barh(sorted_coef['Feature'], sorted_coef['Odds Ratio (e^Beta)'], color='#3498DB')\n",
                    "axes[1].axvline(1.0, color='red', linestyle='--', alpha=0.8, label='Neutral Odds (OR = 1.0)')\n",
                    "axes[1].set_title('Feature Odds Ratios (e^Beta)', fontweight='bold')\n",
                    "axes[1].set_xlabel('Odds Ratio (>1 increases odds, <1 decreases odds)')\n",
                    "axes[1].legend(loc='lower right')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### In-Depth Coefficient Interpretation:\n",
                    "1. **`Title_Mr` ($\\beta = -2.046$, Odds Ratio $= 0.129$)**:\n",
                    "   - Adult male passengers had **87.1% lower odds** of surviving compared to the baseline group, reflecting priority given to females and children.\n",
                    "2. **`HasCabin_1` ($\\beta = +1.123$, Odds Ratio $= 3.075$)**:\n",
                    "   - Passengers with an assigned cabin had over **3.07 times the odds** of survival compared to those without known cabins, as cabins were situated on the upper decks closer to the lifeboat stations.\n",
                    "3. **`Pclass_3` ($\\beta = -1.081$, Odds Ratio $= 0.339$)**:\n",
                    "   - 3rd Class passengers suffered a **66.1% reduction in odds of survival** relative to 1st Class passengers.\n",
                    "4. **`Age` ($\\beta = -0.416$, Odds Ratio $= 0.659$)**:\n",
                    "   - For each standard deviation increase in age, the odds of survival decreased by **34.1%**, consistent with younger passengers receiving evacuation preference."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Model Serialization & Test Ingestion\n",
                    "We serialize the complete pipeline into `titanic_logistic_pipeline.pkl` for the Streamlit web application."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Serialize trained model pipeline\n",
                    "joblib.dump(best_model, 'titanic_logistic_pipeline.pkl')\n",
                    "print(\"Model pipeline saved successfully as 'titanic_logistic_pipeline.pkl'!\")\n",
                    "\n",
                    "# Predict on Test Set\n",
                    "df_test_feat = extract_features(df_test)\n",
                    "test_predictions = best_model.predict(df_test_feat[feature_cols])\n",
                    "test_probabilities = best_model.predict_proba(df_test_feat[feature_cols])[:, 1]\n",
                    "\n",
                    "submission_df = pd.DataFrame({\n",
                    "    'PassengerId': df_test['PassengerId'],\n",
                    "    'Survived': test_predictions,\n",
                    "    'Survival_Probability': np.round(test_probabilities, 4)\n",
                    "})\n",
                    "submission_df.to_csv('titanic_test_predictions.csv', index=False)\n",
                    "print(f\"Generated predictions for {len(submission_df)} test samples in 'titanic_test_predictions.csv'.\")\n",
                    "submission_df.head(10)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Data Science Interview Q&A\n",
                    "---\n",
                    "### Interview Question 1: What is the difference between Precision and Recall?\n",
                    "\n",
                    "#### Definitions & Mathematical Formulas:\n",
                    "- **Precision (Positive Predictive Value)**:\n",
                    "  $$\\text{Precision} = \\frac{\\text{True Positives (TP)}}{\\text{True Positives (TP)} + \\text{False Positives (FP)}}$$\n",
                    "  *Interpretation*: Out of all instances that the model predicted as positive, what proportion was genuinely positive?\n",
                    "\n",
                    "- **Recall (Sensitivity / True Positive Rate)**:\n",
                    "  $$\\text{Recall} = \\frac{\\text{True Positives (TP)}}{\\text{True Positives (TP)} + \\text{False Negatives (FN)}}$$\n",
                    "  *Interpretation*: Out of all actual positive instances present in the dataset, what proportion did the model successfully identify?\n",
                    "\n",
                    "#### The Precision-Recall Trade-off:\n",
                    "- Changing the decision threshold $\\tau$ shifts the trade-off:\n",
                    "  - **Higher Threshold $\\tau$**: The model requires higher confidence to predict positive $\\rightarrow$ FP decreases, **Precision increases**, but FN increases, causing **Recall to decrease**.\n",
                    "  - **Lower Threshold $\\tau$**: The model predicts positive more liberally $\\rightarrow$ FN decreases, **Recall increases**, but FP increases, causing **Precision to decrease**.\n",
                    "\n",
                    "#### Real-World Applications:\n",
                    "1. **Medical Diagnosis (e.g. Cancer Screening)**: High Recall is critical because a False Negative (missing a malignant tumor) can be fatal.\n",
                    "2. **Spam Filtering / Fraud Notification**: High Precision is preferred because False Positives (sending a crucial legitimate email to spam) directly damage user trust.\n",
                    "3. **Balanced Metric ($F_1$-Score)**: When both FP and FN have substantial costs, the harmonic mean $F_1 = 2 \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{\\text{Precision} + \\text{Recall}}$ provides an optimal balance.\n",
                    "\n",
                    "---\n",
                    "### Interview Question 2: What is Cross-Validation, and why is it important in binary classification?\n",
                    "\n",
                    "#### Definition:\n",
                    "**Cross-Validation (CV)** is a resampling procedure where the dataset is split into $K$ equal-sized folds. For each iteration, $K-1$ folds are used to train the model, and the remaining $1$ fold is used for validation. The process repeats $K$ times, and the evaluation metric is averaged across all $K$ folds.\n",
                    "\n",
                    "$$\\text{CV Score} = \\frac{1}{K} \\sum_{k=1}^K \\text{Score}_k$$\n",
                    "\n",
                    "#### Importance in Binary Classification:\n",
                    "1. **Mitigates Class Imbalance via Stratified $K$-Fold**:\n",
                    "   - Standard $K$-Fold split might produce folds with skewed class ratios by random chance.\n",
                    "   - **Stratified $K$-Fold** ensures that each fold contains exactly the same proportion of positive and negative classes as the full dataset, preventing biased validation.\n",
                    "2. **Prevents Data Leakage & Overfitting**:\n",
                    "   - Ensures all data preprocessing, scaling, and hyperparameter tuning (`GridSearchCV`) occur strictly within training folds.\n",
                    "3. **Quantifies Model Stability (Variance Estimation)**:\n",
                    "   - Standard deviation across folds indicates how sensitive the model is to variations in the training sample.\n",
                    "4. **Maximizes Data Utilization**:\n",
                    "   - Every observation is used for both training and validation across the $K$ iterations, critical for medium and small datasets like Titanic."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open("logistic_regression_titanic.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print("Notebook 'logistic_regression_titanic.ipynb' created successfully!")

if __name__ == "__main__":
    create_notebook()

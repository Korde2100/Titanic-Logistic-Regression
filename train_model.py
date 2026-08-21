"""
Titanic Survival Prediction - Logistic Regression Pipeline
==========================================================
End-to-end Machine Learning script covering:
1. Data Ingestion & EDA Visualizations
2. Feature Engineering & Preprocessing Pipeline
3. Model Training & Hyperparameter Tuning via Stratified K-Fold Cross Validation
4. Comprehensive Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
5. Coefficient & Odds Ratio Interpretation
6. Model Serialization for Streamlit Web Application
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)

# Set styling for plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

def extract_features(df_raw):
    """
    Extract engineered features from raw Titanic dataframe.
    """
    df = df_raw.copy()
    
    # 1. Title Extraction
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        # Group rare titles
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
            'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
            'Capt': 'Rare', 'Sir': 'Rare'
        }
        df['Title'] = df['Title'].map(title_mapping).fillna('Rare')
    else:
        df['Title'] = 'Mr'
        
    # 2. Family Size & IsAlone
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    else:
        df['FamilySize'] = 1
        df['IsAlone'] = 1
        
    # 3. Cabin Deck Indicator
    if 'Cabin' in df.columns:
        df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
    else:
        df['HasCabin'] = 0
        
    return df

def generate_eda_plots(df, output_dir='plots'):
    """
    Generate and save comprehensive EDA visualization figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_eda = extract_features(df)
    
    # --- 1. Demographics vs Survival ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Sex vs Survival
    sns.countplot(data=df_eda, x='Sex', hue='Survived', palette=['#E74C3C', '#2ECC71'], ax=axes[0])
    axes[0].set_title('Survival Count by Sex', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Sex')
    axes[0].set_ylabel('Passenger Count')
    axes[0].legend(['Perished (0)', 'Survived (1)'])
    
    # Pclass vs Survival
    sns.countplot(data=df_eda, x='Pclass', hue='Survived', palette=['#E74C3C', '#2ECC71'], ax=axes[1])
    axes[1].set_title('Survival Count by Passenger Class', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Pclass (1 = 1st, 2 = 2nd, 3 = 3rd)')
    axes[1].set_ylabel('Passenger Count')
    axes[1].legend(['Perished (0)', 'Survived (1)'])
    
    # Embarked vs Survival
    sns.countplot(data=df_eda, x='Embarked', hue='Survived', palette=['#E74C3C', '#2ECC71'], ax=axes[2])
    axes[2].set_title('Survival Count by Embarkation Port', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Embarked (C = Cherbourg, Q = Queenstown, S = Southampton)')
    axes[2].set_ylabel('Passenger Count')
    axes[2].legend(['Perished (0)', 'Survived (1)'])
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'eda_survival_demographics.png'), dpi=300)
    plt.close()
    
    # --- 2. Age and Fare Distributions ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Age Distribution
    sns.histplot(data=df_eda, x='Age', hue='Survived', kde=True, bins=30, palette=['#E74C3C', '#2ECC71'], element='step', ax=axes[0, 0])
    axes[0, 0].set_title('Age Distribution by Survival Status', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Age (Years)')
    
    # Age Boxplot by Pclass
    sns.boxplot(data=df_eda, x='Pclass', y='Age', hue='Survived', palette=['#E74C3C', '#2ECC71'], ax=axes[0, 1])
    axes[0, 1].set_title('Age by Passenger Class and Survival', fontsize=13, fontweight='bold')
    
    # Fare Distribution
    sns.histplot(data=df_eda, x='Fare', hue='Survived', kde=True, bins=30, palette=['#E74C3C', '#2ECC71'], element='step', ax=axes[1, 0])
    axes[1, 0].set_title('Fare Distribution by Survival Status', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Fare ($)')
    
    # Fare Boxplot by Pclass
    sns.boxplot(data=df_eda, x='Pclass', y='Fare', hue='Survived', palette=['#E74C3C', '#2ECC71'], ax=axes[1, 1])
    axes[1, 1].set_title('Fare by Class and Survival (Capped View)', fontsize=13, fontweight='bold')
    axes[1, 1].set_ylim(0, 300)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'eda_age_fare_distribution.png'), dpi=300)
    plt.close()
    
    # --- 3. Family Effects & Title ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Family Size vs Survival Rate
    fam_surv = df_eda.groupby('FamilySize')['Survived'].mean().reset_index()
    sns.barplot(data=fam_surv, x='FamilySize', y='Survived', palette='Blues_d', ax=axes[0])
    axes[0].set_title('Survival Rate by Family Size', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Survival Rate')
    axes[0].set_ylim(0, 1)
    
    # IsAlone vs Survival Rate
    sns.barplot(data=df_eda, x='IsAlone', y='Survived', palette=['#3498DB', '#9B59B6'], ax=axes[1])
    axes[1].set_title('Survival Rate: Alone vs With Family', fontsize=14, fontweight='bold')
    axes[1].set_xticklabels(['With Family (0)', 'Alone (1)'])
    axes[1].set_ylabel('Survival Rate')
    axes[1].set_ylim(0, 1)
    
    # Title vs Survival Rate
    title_surv = df_eda.groupby('Title')['Survived'].mean().reset_index().sort_values(by='Survived', ascending=False)
    sns.barplot(data=title_surv, x='Title', y='Survived', palette='viridis', ax=axes[2])
    axes[2].set_title('Survival Rate by Extracted Title', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Survival Rate')
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'eda_family_effects.png'), dpi=300)
    plt.close()
    
    # --- 4. Correlation Heatmap ---
    plt.figure(figsize=(10, 8))
    numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'HasCabin']
    corr_matrix = df_eda[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', mask=mask, cbar_kws={'shrink': .8}, linewidths=0.5)
    plt.title('Correlation Matrix of Numeric & Engineered Features', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_correlation_heatmap.png'), dpi=300)
    plt.close()
    
    print(f"[+] EDA plots generated and saved to '{output_dir}/'")

def build_and_train_pipeline(train_path='Titanic_train.csv'):
    """
    Build scikit-learn preprocessing ColumnTransformer and LogisticRegression pipeline.
    Train with hyperparameter tuning and Stratified Cross-Validation.
    """
    # Load dataset
    df = pd.read_csv(train_path)
    df_feat = extract_features(df)
    
    # Define features and target
    target = 'Survived'
    y = df_feat[target]
    
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'HasCabin']
    X = df_feat[feature_cols]
    
    # Stratified Train/Validation Split (80% Train, 20% Validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Numeric & Categorical columns
    numeric_features = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch']
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'HasCabin']
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Full Model Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Hyperparameter Grid
    param_grid = {
        'classifier__C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs', 'liblinear'],
        'classifier__class_weight': [None, 'balanced']
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, return_train_score=True
    )
    
    print("[*] Performing 5-Fold Stratified Cross Validation & Grid Search...")
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_auc = grid_search.best_score_
    
    print(f"[+] Best CV ROC-AUC: {best_cv_auc:.4f}")
    print(f"[+] Best Hyperparameters: {best_params}")
    
    # Cross validation detailed scores on best model
    cv_results = cross_validate(
        best_pipeline, X_train, y_train, cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )
    
    cv_metrics = {
        'accuracy_mean': float(np.mean(cv_results['test_accuracy'])),
        'accuracy_std': float(np.std(cv_results['test_accuracy'])),
        'precision_mean': float(np.mean(cv_results['test_precision'])),
        'recall_mean': float(np.mean(cv_results['test_recall'])),
        'f1_mean': float(np.mean(cv_results['test_f1'])),
        'roc_auc_mean': float(np.mean(cv_results['test_roc_auc'])),
    }
    
    # Evaluate on Validation Set (Out-of-sample)
    y_pred = best_pipeline.predict(X_val)
    y_prob = best_pipeline.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    cm = confusion_matrix(y_val, y_pred)
    cr = classification_report(y_val, y_pred, output_dict=True)
    
    eval_metrics = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(auc),
        'confusion_matrix': cm.tolist(),
        'cv_metrics': cv_metrics,
        'best_params': best_params
    }
    
    print("\n" + "="*50)
    print("        VALIDATION EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:\n", classification_report(y_val, y_pred))
    
    # Extract Feature Names & Coefficients
    preprocessor_step = best_pipeline.named_steps['preprocessor']
    classifier_step = best_pipeline.named_steps['classifier']
    
    # Get feature names after one-hot encoding
    cat_encoder = preprocessor_step.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_feature_names
    
    coefficients = classifier_step.coef_[0]
    intercept = classifier_step.intercept_[0]
    odds_ratios = np.exp(coefficients)
    
    coef_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': coefficients,
        'Odds_Ratio': odds_ratios,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values(by='Abs_Coefficient', ascending=False).reset_index(drop=True)
    
    print("\n" + "="*70)
    print("       LOGISTIC REGRESSION COEFFICIENTS & ODDS RATIOS")
    print("="*70)
    print(f"Intercept: {intercept:.4f} (Baseline Log-Odds)")
    print(coef_df[['Feature', 'Coefficient', 'Odds_Ratio']].to_string(index=False))
    
    return best_pipeline, X_train, X_val, y_train, y_val, y_prob, eval_metrics, coef_df, intercept

def generate_evaluation_plots(y_val, y_prob, y_pred, coef_df, output_dir='plots'):
    """
    Generate ROC Curve, Confusion Matrix, PR Curve, and Feature Importance/Odds Ratios.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Confusion Matrix Plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Perished (0)', 'Survived (1)'],
                yticklabels=['Perished (0)', 'Survived (1)'], ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title('Validation Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'model_confusion_matrix.png'), dpi=300)
    plt.close()
    
    # --- 2. ROC Curve ---
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)
    auc_score = roc_auc_score(y_val, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#2980B9', lw=2.5, label=f'Logistic Regression (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], color='#7F8C8D', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.500)')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#2980B9')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'model_roc_curve.png'), dpi=300)
    plt.close()
    
    # --- 3. Precision-Recall Curve ---
    precision_pts, recall_pts, _ = precision_recall_curve(y_val, y_prob)
    avg_prec = average_precision_score(y_val, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_pts, precision_pts, color='#27AE60', lw=2.5, label=f'PR Curve (AP = {avg_prec:.3f})')
    ax.fill_between(recall_pts, precision_pts, alpha=0.15, color='#27AE60')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'model_precision_recall_curve.png'), dpi=300)
    plt.close()
    
    # --- 4. Model Coefficients & Odds Ratios Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Top 12 Coefficients
    plot_df = coef_df.head(12).sort_values(by='Coefficient')
    colors = ['#2ECC71' if c > 0 else '#E74C3C' for c in plot_df['Coefficient']]
    axes[0].barh(plot_df['Feature'], plot_df['Coefficient'], color=colors)
    axes[0].axvline(0, color='black', linestyle='--', alpha=0.7)
    axes[0].set_title('Top Model Coefficients (Log-Odds Impact)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Coefficient Value ($\\beta$)')
    
    # Odds Ratios
    plot_df_or = coef_df.head(12).sort_values(by='Odds_Ratio')
    axes[1].barh(plot_df_or['Feature'], plot_df_or['Odds_Ratio'], color='#3498DB')
    axes[1].axvline(1.0, color='red', linestyle='--', alpha=0.8, label='Neutral Odds (OR = 1.0)')
    axes[1].set_title('Top Features by Odds Ratio ($e^\\beta$)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Odds Ratio ($>1$ increases odds, $<1$ decreases odds)')
    axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'model_coefficients_odds_ratios.png'), dpi=300)
    plt.close()
    
    print(f"[+] Evaluation plots generated and saved to '{output_dir}/'")

def main():
    print("="*60)
    print("   TITANIC LOGISTIC REGRESSION - END-TO-END WORKFLOW")
    print("="*60)
    
    train_path = 'Titanic_train.csv'
    test_path = 'Titanic_test.csv'
    
    if not os.path.exists(train_path):
        train_path = os.path.join('Logistic Regression', 'Titanic_train.csv')
    if not os.path.exists(test_path):
        test_path = os.path.join('Logistic Regression', 'Titanic_test.csv')
        
    df_train = pd.read_csv(train_path)
    
    # 1. EDA Visualizations
    generate_eda_plots(df_train, output_dir='plots')
    
    # 2. Pipeline Training & Evaluation
    best_pipeline, X_train, X_val, y_train, y_val, y_prob, eval_metrics, coef_df, intercept = build_and_train_pipeline(train_path)
    
    # 3. Model Diagnostic Plots
    y_pred = (y_prob >= 0.5).astype(int)
    generate_evaluation_plots(y_val, y_prob, y_pred, coef_df, output_dir='plots')
    
    # 4. Save Model Artifacts
    joblib.dump(best_pipeline, 'titanic_logistic_pipeline.pkl')
    print("\n[+] Trained pipeline model saved to 'titanic_logistic_pipeline.pkl'")
    
    coef_df.to_csv('model_coefficients.csv', index=False)
    with open('model_metrics.json', 'w') as f:
        json.dump(eval_metrics, f, indent=4)
    print("[+] Model metrics and coefficients saved.")
    
    # 5. Predict on Test Set if available
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        df_test_feat = extract_features(df_test)
        feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'HasCabin']
        test_preds = best_pipeline.predict(df_test_feat[feature_cols])
        test_probs = best_pipeline.predict_proba(df_test_feat[feature_cols])[:, 1]
        
        submission = pd.DataFrame({
            'PassengerId': df_test['PassengerId'],
            'Survived': test_preds,
            'Survival_Probability': np.round(test_probs, 4)
        })
        submission.to_csv('titanic_test_predictions.csv', index=False)
        print(f"[+] Predictions generated for {len(submission)} test samples -> 'titanic_test_predictions.csv'")

    print("\n" + "="*60)
    print("   TRAINING & EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == '__main__':
    main()

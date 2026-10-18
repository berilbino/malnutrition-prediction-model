"""
TRAIN 3-STAGE MACHINE LEARNING PIPELINE
1. Model 1: Child Survival / Mortality Risk (Binary Classification)
2. Model 2: Malnutrition Status Severity (Multi-class: Normal, MAM, SAM)
3. Model 3: Disease Susceptibility (Multi-class: Healthy, ARI, Diarrheal, Febrile)

Saves trained models and evaluation metrics to the models/ directory.
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_path = os.path.join(project_dir, 'data', 'unicef_labeled_data.csv')
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70)
    print("TRAINING 3-STAGE CASCADING MACHINE LEARNING PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} records.\n")

    # Features selection
    categorical_cols = [
        'Gender', 'Region', 'Exclusive_BF_6M', 'Safe_Water', 'Sanitation',
        'Vaccinated', 'Recent_Illness', 'Household_Income', 'Mother_Education',
        'Food_Poverty', 'Residence', 'Preterm_Birth', 'Vitamin_A_Supplement',
        'Diarrhea_Last_2Weeks', 'Fever_Last_2Weeks', 'Cough_Last_2Weeks', 'ARI_Symptoms'
    ]
    numeric_cols = [
        'Age_Months', 'Birth_Weight_kg', 'Dietary_Diversity', 'Meal_Frequency',
        'Household_Size', 'Mother_Age', 'Mother_BMI'
    ]

    X = df[categorical_cols + numeric_cols]
    y_survival = df['Survival_Status']
    y_malnutrition = df['Malnutrition_Status']
    y_disease = df['Disease_Status']

    # 2. Build Preprocessing Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', StandardScaler(), numeric_cols)
        ]
    )

    # Split dataset (80% train, 20% test)
    X_train, X_test, y_surv_train, y_surv_test, y_mal_train, y_mal_test, y_dis_train, y_dis_test = train_test_split(
        X, y_survival, y_malnutrition, y_disease, test_size=0.2, random_state=42, stratify=y_survival
    )

    metrics_report = {}

    # ----------------------------------------------------
    # STAGE 1: SURVIVAL PREDICTION MODEL
    # ----------------------------------------------------
    print(">>> 1/3 Training Stage 1: Survival Classifier (Mortality Risk)...")
    pipe_survival = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1, class_weight='balanced'))
    ])
    pipe_survival.fit(X_train, y_surv_train)
    y_surv_pred = pipe_survival.predict(X_test)
    acc_surv = accuracy_score(y_surv_test, y_surv_pred)
    print(f"    Survival Model Accuracy: {acc_surv*100:.2f}%")
    print(classification_report(y_surv_test, y_surv_pred, target_names=['Survives', 'Critical Risk']))

    joblib.dump(pipe_survival, os.path.join(models_dir, 'survival_model.pkl'))
    metrics_report['survival'] = {
        'accuracy': round(acc_surv, 4),
        'confusion_matrix': confusion_matrix(y_surv_test, y_surv_pred).tolist(),
        'target_names': ['Survives', 'Critical / High Mortality Risk']
    }

    # ----------------------------------------------------
    # STAGE 2: MALNUTRITION STATUS MODEL
    # ----------------------------------------------------
    print("\n>>> 2/3 Training Stage 2: Malnutrition Classifier (Normal vs MAM vs SAM)...")
    pipe_malnutrition = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=14, random_state=42, n_jobs=-1))
    ])
    pipe_malnutrition.fit(X_train, y_mal_train)
    y_mal_pred = pipe_malnutrition.predict(X_test)
    acc_mal = accuracy_score(y_mal_test, y_mal_pred)
    print(f"    Malnutrition Model Accuracy: {acc_mal*100:.2f}%")
    print(classification_report(y_mal_test, y_mal_pred, target_names=['Normal', 'MAM', 'SAM']))

    joblib.dump(pipe_malnutrition, os.path.join(models_dir, 'malnutrition_model.pkl'))
    metrics_report['malnutrition'] = {
        'accuracy': round(acc_mal, 4),
        'confusion_matrix': confusion_matrix(y_mal_test, y_mal_pred).tolist(),
        'target_names': ['Normal / Well-Nourished', 'Moderate Malnutrition (MAM)', 'Severe Malnutrition (SAM)']
    }

    # ----------------------------------------------------
    # STAGE 3: DISEASE PREDICTION MODEL
    # ----------------------------------------------------
    print("\n>>> 3/3 Training Stage 3: Disease Prediction Classifier...")
    pipe_disease = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=14, random_state=42, n_jobs=-1))
    ])
    pipe_disease.fit(X_train, y_dis_train)
    y_dis_pred = pipe_disease.predict(X_test)
    acc_dis = accuracy_score(y_dis_test, y_dis_pred)
    print(f"    Disease Model Accuracy: {acc_dis*100:.2f}%")
    print(classification_report(y_dis_test, y_dis_pred, target_names=['Healthy', 'ARI', 'Diarrhea', 'Febrile']))

    joblib.dump(pipe_disease, os.path.join(models_dir, 'disease_model.pkl'))
    metrics_report['disease'] = {
        'accuracy': round(acc_dis, 4),
        'confusion_matrix': confusion_matrix(y_dis_test, y_dis_pred).tolist(),
        'target_names': ['Healthy / No Acute Infection', 'Acute Respiratory Infection (ARI/Pneumonia)', 'Diarrheal / Waterborne Illness', 'Febrile / Systemic Illness']
    }

    # Save feature names and metadata
    metadata = {
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'metrics': metrics_report
    }
    with open(os.path.join(models_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    print("\n" + "=" * 70)
    print("ALL 3 MODELS SUCCESSFULLY TRAINED AND SAVED TO 'models/' DIRECTORY!")
    print("=" * 70)

if __name__ == '__main__':
    main()

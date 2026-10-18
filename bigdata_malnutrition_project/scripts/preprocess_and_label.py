import pandas as pd
import numpy as np

def preprocess_and_label(input_csv, output_csv):
    print(f"Loading raw dataset from: {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Dataset loaded: {df.shape[0]:,} rows and {df.shape[1]} columns.")

    # 1. Clean missing values
    # Mother_Education has ~12% missing values, impute with mode
    mode_edu = df['Mother_Education'].mode()[0]
    df['Mother_Education'] = df['Mother_Education'].fillna(mode_edu)

    # Convert binary flags to 1/0 helpers for scoring
    def to_bin(series):
        return (series.astype(str).str.strip().str.lower() == 'yes').astype(int)

    food_pov = to_bin(df['Food_Poverty'])
    safe_water_no = 1 - to_bin(df['Safe_Water'])
    sanitation_no = 1 - to_bin(df['Sanitation'])
    unvaccinated = 1 - to_bin(df['Vaccinated'])
    no_bf = 1 - to_bin(df['Exclusive_BF_6M'])
    preterm = to_bin(df['Preterm_Birth'])
    no_vit_a = 1 - to_bin(df['Vitamin_A_Supplement'])
    diarrhea = to_bin(df['Diarrhea_Last_2Weeks'])
    fever = to_bin(df['Fever_Last_2Weeks'])
    cough = to_bin(df['Cough_Last_2Weeks'])
    ari = to_bin(df['ARI_Symptoms'])
    recent_ill = to_bin(df['Recent_Illness'])
    low_income = (df['Household_Income'] == 'Low').astype(int)

    # 2. Derive Malnutrition Status (WHO / UNICEF Composite Risk Index)
    # Risk points based on clinical guidelines
    mal_score = (
        food_pov * 2.0 +
        (df['Dietary_Diversity'] <= 2).astype(int) * 2.0 +
        ((df['Dietary_Diversity'] > 2) & (df['Dietary_Diversity'] <= 4)).astype(int) * 1.0 +
        (df['Meal_Frequency'] <= 2).astype(int) * 1.0 +
        (df['Birth_Weight_kg'] < 2.5).astype(int) * 2.0 +
        (df['Mother_BMI'] < 18.5).astype(int) * 1.5 +
        no_bf * 1.0 +
        safe_water_no * 0.75 +
        sanitation_no * 0.75 +
        low_income * 0.5
    )

    df['Malnutrition_Score'] = mal_score.round(2)
    # Categories:
    # 0 = Normal / Well-Nourished
    # 1 = Moderate Acute Malnutrition (MAM)
    # 2 = Severe Acute Malnutrition (SAM)
    conditions_mal = [
        mal_score >= 5.5,
        (mal_score >= 3.0) & (mal_score < 5.5)
    ]
    choices_mal = [2, 1]
    df['Malnutrition_Status'] = np.select(conditions_mal, choices_mal, default=0)
    
    mal_labels = {0: 'Normal / Well-Nourished', 1: 'Moderate Malnutrition (MAM)', 2: 'Severe Malnutrition (SAM)'}
    df['Malnutrition_Label'] = df['Malnutrition_Status'].map(mal_labels)

    # 3. Derive Survival / Under-5 Mortality Risk (Stage 1 Target)
    # Risk accumulates from extreme LBW, preterm birth, SAM, lack of vaccination, and co-occurring infections
    symptom_count = diarrhea + fever + cough + ari
    mortality_risk_score = (
        (df['Malnutrition_Status'] == 2).astype(int) * 3.0 +
        (df['Birth_Weight_kg'] < 2.3).astype(int) * 2.5 +
        preterm * 2.0 +
        unvaccinated * 2.0 +
        (df['Age_Months'] <= 12).astype(int) * 1.5 +
        (symptom_count >= 2).astype(int) * 2.5 +
        safe_water_no * 1.0
    )

    # High mortality risk (1 = Critical / Fatal Risk, 0 = Survives)
    df['Mortality_Risk_Score'] = mortality_risk_score.round(2)
    df['Survival_Status'] = np.where(mortality_risk_score >= 7.5, 1, 0)
    surv_labels = {0: 'Survives', 1: 'Critical / High Mortality Risk'}
    df['Survival_Label'] = df['Survival_Status'].map(surv_labels)

    # 4. Derive Disease Outcome (Stage 3 Target)
    # Clinical syndromic surveillance mapping:
    # 0 = Healthy / No Severe Infection
    # 1 = Acute Respiratory Infection (ARI / Pneumonia)
    # 2 = Diarrheal / Waterborne Infection
    # 3 = Febrile Illness / Systemic Infection
    conditions_disease = [
        (ari == 1) & (cough == 1),
        (diarrhea == 1),
        (fever == 1)
    ]
    choices_disease = [1, 2, 3]
    df['Disease_Status'] = np.select(conditions_disease, choices_disease, default=0)
    disease_labels = {
        0: 'Healthy / No Acute Infection',
        1: 'Acute Respiratory Infection (ARI/Pneumonia)',
        2: 'Diarrheal / Waterborne Illness',
        3: 'Febrile / Systemic Illness'
    }
    df['Disease_Label'] = df['Disease_Status'].map(disease_labels)

    # Save labeled dataset
    df.to_csv(output_csv, index=False)
    print(f"\nSuccessfully created preprocessed & labeled dataset: {output_csv}")
    print("\n--- TARGET DISTRIBUTIONS SUMMARY ---")
    print("\n1. Stage 1 - Survival Status:")
    print(df['Survival_Label'].value_counts(normalize=True).mul(100).round(2).astype(str) + '% (' + df['Survival_Label'].value_counts().astype(str) + ')')

    print("\n2. Stage 2 - Malnutrition Status:")
    print(df['Malnutrition_Label'].value_counts(normalize=True).mul(100).round(2).astype(str) + '% (' + df['Malnutrition_Label'].value_counts().astype(str) + ')')

    print("\n3. Stage 3 - Disease Outcome:")
    print(df['Disease_Label'].value_counts(normalize=True).mul(100).round(2).astype(str) + '% (' + df['Disease_Label'].value_counts().astype(str) + ')')

if __name__ == '__main__':
    in_file = r"C:\Users\beril\.gemini\antigravity-ide\scratch\bigdata_malnutrition_project\data\unicef_dataset.csv"
    out_file = r"C:\Users\beril\.gemini\antigravity-ide\scratch\bigdata_malnutrition_project\data\unicef_labeled_data.csv"
    preprocess_and_label(in_file, out_file)

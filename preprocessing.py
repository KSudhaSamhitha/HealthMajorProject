# preprocessing_final.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "datasets/"
OUTPUT_PATH = "processed_data/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# -----------------------------
# Load Datasets
# -----------------------------
print("="*60)
print("LOADING DATASETS")
print("="*60)
heart_df = pd.read_csv(DATA_PATH + "heart.csv")
frame_df = pd.read_csv(DATA_PATH + "framingham.csv")

print(f"Heart dataset: {heart_df.shape} - Columns: {list(heart_df.columns)}")
print(f"Framingham dataset: {frame_df.shape} - Columns: {list(frame_df.columns)}")

# -----------------------------
# Rename Target Columns (Unify)
# -----------------------------
heart_df.rename(columns={"target": "heart_disease"}, inplace=True)
frame_df.rename(columns={"TenYearCHD": "heart_disease"}, inplace=True)

# -----------------------------
# Handle Missing Values
# -----------------------------
def handle_missing_values(df):
    for col in df.columns:
        if df[col].dtype != "object":
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

print("\n" + "="*60)
print("HANDLING MISSING VALUES")
print("="*60)
heart_df = handle_missing_values(heart_df)
frame_df = handle_missing_values(frame_df)
print("✅ Missing values handled")

# -----------------------------
# Feature Engineering & Mapping
# -----------------------------
print("\n" + "="*60)
print("FEATURE MAPPING AND ENGINEERING")
print("="*60)

# Create unified feature set
unified_data = []

# Process Heart Dataset
print("\n🫀 Processing Heart Dataset...")
heart_processed = pd.DataFrame()
heart_processed['age'] = heart_df['age']
heart_processed['sex'] = heart_df['sex']  # 1=male, 0=female
heart_processed['chest_pain_type'] = heart_df['cp']  # 0-3
heart_processed['resting_bp'] = heart_df['trestbps']
heart_processed['cholesterol'] = heart_df['chol']
heart_processed['fasting_blood_sugar'] = heart_df['fbs']  # 1 if >120 mg/dl
heart_processed['resting_ecg'] = heart_df['restecg']  # 0-2
heart_processed['max_heart_rate'] = heart_df['thalach']
heart_processed['exercise_angina'] = heart_df['exang']  # 1=yes, 0=no
heart_processed['st_depression'] = heart_df['oldpeak']
heart_processed['st_slope'] = heart_df['slope']  # 0-2
heart_processed['num_vessels'] = heart_df['ca']  # 0-3
heart_processed['thalassemia'] = heart_df['thal']  # 1-3
heart_processed['dataset'] = 'heart'
heart_processed['heart_disease'] = heart_df['heart_disease']
print(f"  ✅ Created {len(heart_processed.columns)} features for {len(heart_processed)} samples")

# Process Framingham Dataset
print("\n🫀 Processing Framingham Dataset...")
frame_processed = pd.DataFrame()
frame_processed['age'] = frame_df['age']
frame_processed['sex'] = frame_df['male']  # 1=male, 0=female
frame_processed['education'] = frame_df['education']
frame_processed['current_smoker'] = frame_df['currentSmoker']
frame_processed['cigs_per_day'] = frame_df['cigsPerDay']
frame_processed['bp_meds'] = frame_df['BPMeds']
frame_processed['prevalent_stroke'] = frame_df['prevalentStroke']
frame_processed['prevalent_hypertension'] = frame_df['prevalentHyp']
frame_processed['diabetes'] = frame_df['diabetes']
frame_processed['total_cholesterol'] = frame_df['totChol']
frame_processed['sys_bp'] = frame_df['sysBP']
frame_processed['dia_bp'] = frame_df['diaBP']
frame_processed['bmi'] = frame_df['BMI']
frame_processed['heart_rate'] = frame_df['heartRate']
frame_processed['glucose'] = frame_df['glucose']
frame_processed['dataset'] = 'framingham'
frame_processed['heart_disease'] = frame_df['heart_disease']
print(f"  ✅ Created {len(frame_processed.columns)} features for {len(frame_processed)} samples")

# -----------------------------
# Combine Datasets
# -----------------------------
print("\n" + "="*60)
print("COMBINING DATASETS")
print("="*60)

# Concatenate
combined_df = pd.concat([heart_processed, frame_processed], axis=0, ignore_index=True)
print(f"Combined dataset shape: {combined_df.shape}")
print(f"Combined columns: {list(combined_df.columns)}")

# Check class distribution
print("\n📊 Class Distribution:")
print(combined_df['heart_disease'].value_counts())
print(f"Percentage with heart disease: {combined_df['heart_disease'].mean()*100:.2f}%")

# -----------------------------
# Handle Missing Values in Combined Dataset
# -----------------------------
print("\n" + "="*60)
print("FINAL MISSING VALUE HANDLING")
print("="*60)

# Check missing values
missing = combined_df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0])

# Fill missing values
for col in combined_df.columns:
    if combined_df[col].dtype != "object" and col != 'heart_disease' and col != 'dataset':
        combined_df[col].fillna(combined_df[col].median(), inplace=True)

print("✅ Missing values handled")

# -----------------------------
# Feature Selection
# -----------------------------
print("\n" + "="*60)
print("FEATURE SELECTION")
print("="*60)

# Select features for modeling (exclude metadata)
feature_cols = [col for col in combined_df.columns 
                if col not in ['heart_disease', 'dataset']]

print(f"Selected {len(feature_cols)} features for modeling:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i}. {col}")

# -----------------------------
# Prepare Final Dataset
# -----------------------------
X = combined_df[feature_cols]
y = combined_df['heart_disease']

print(f"\n✅ Final feature matrix shape: {X.shape}")
print(f"✅ Target vector shape: {y.shape}")

# -----------------------------
# Feature Scaling
# -----------------------------
print("\n" + "="*60)
print("FEATURE SCALING")
print("="*60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("✅ Features scaled using StandardScaler")

# -----------------------------
# Save Processed Data
# -----------------------------
print("\n" + "="*60)
print("SAVING PROCESSED DATA")
print("="*60)

X_scaled_df.to_csv(OUTPUT_PATH + "X_scaled.csv", index=False)
y.to_csv(OUTPUT_PATH + "y.csv", index=False)

print(f"✅ X_scaled.csv saved with shape: {X_scaled_df.shape}")
print(f"✅ y.csv saved with shape: {y.shape}")

print("\n" + "="*60)
print("🎉 PREPROCESSING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\nFinal dataset contains {len(feature_cols)} features:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i}. {col}")
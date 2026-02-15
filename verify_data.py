# verify_data.py
import pandas as pd
import numpy as np

print("="*60)
print("VERIFYING PROCESSED DATA")
print("="*60)

# Load the processed data
X = pd.read_csv("processed_data/X_scaled.csv")
y = pd.read_csv("processed_data/y.csv")

print(f"\n📊 X shape: {X.shape}")
print(f"📊 y shape: {y.shape}")

print(f"\n📋 Features ({len(X.columns)} total):")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")

print(f"\n📈 Feature statistics:")
print(X.describe().round(3))

print(f"\n🎯 Target distribution:")
print(y['heart_disease'].value_counts())
print(f"  Percentage with heart disease: {y['heart_disease'].mean()*100:.2f}%")

print("\n✅ Verification complete!")
# debug_datasets.py
import pandas as pd

print("="*60)
print("DEBUGGING DATASETS")
print("="*60)

# Load datasets
heart_df = pd.read_csv("datasets/heart.csv")
frame_df = pd.read_csv("datasets/framingham.csv")

print("\n📊 HEART DATASET")
print(f"Shape: {heart_df.shape}")
print(f"Columns: {heart_df.columns.tolist()}")
print(f"\nFirst few rows:")
print(heart_df.head())

print("\n" + "="*60)
print("\n📊 FRAMINGHAM DATASET")
print(f"Shape: {frame_df.shape}")
print(f"Columns: {frame_df.columns.tolist()}")
print(f"\nFirst few rows:")
print(frame_df.head())

# Check common columns
heart_cols = set(heart_df.columns)
frame_cols = set(frame_df.columns)
common_cols = heart_cols.intersection(frame_cols)

print("\n" + "="*60)
print(f"\n🔍 Common columns between datasets: {common_cols}")
print(f"Number of common columns: {len(common_cols)}")
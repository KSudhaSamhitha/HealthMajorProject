import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "datasets/"
OUTPUT_PATH = "processed_data/"

import os
os.makedirs(OUTPUT_PATH, exist_ok=True)

# -----------------------------
# Load Datasets
# -----------------------------
heart_df = pd.read_csv(DATA_PATH + "heart.csv")
frame_df = pd.read_csv(DATA_PATH + "framingham.csv")

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

heart_df = handle_missing_values(heart_df)
frame_df = handle_missing_values(frame_df)

# -----------------------------
# Align Common Features
# -----------------------------
common_features = list(set(heart_df.columns) & set(frame_df.columns))
common_features.remove("heart_disease")

heart_common = heart_df[common_features + ["heart_disease"]]
frame_common = frame_df[common_features + ["heart_disease"]]

# -----------------------------
# Multimodal Dataset Fusion
# (Vertical Concatenation)
# -----------------------------
multimodal_df = pd.concat(
    [heart_common, frame_common],
    axis=0,
    ignore_index=True
)

print("Multimodal Dataset Shape:", multimodal_df.shape)

# -----------------------------
# Feature / Target Split
# -----------------------------
X = multimodal_df.drop("heart_disease", axis=1)
y = multimodal_df["heart_disease"]

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# -----------------------------
# Save Processed Data
# -----------------------------
X_scaled_df.to_csv(OUTPUT_PATH + "X_scaled.csv", index=False)
y.to_csv(OUTPUT_PATH + "y.csv", index=False)

print("✅ Preprocessing & Multimodal Fusion Completed")
print("Saved files:")
print(" - processed_data/X_scaled.csv")
print(" - processed_data/y.csv")

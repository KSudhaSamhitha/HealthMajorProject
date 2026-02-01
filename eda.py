import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use("seaborn-v0_8")

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "datasets/"
PLOT_PATH = "outputs/plots/"
os.makedirs(PLOT_PATH, exist_ok=True)

# -----------------------------
# Load Datasets
# -----------------------------
heart_df = pd.read_csv(DATA_PATH + "heart.csv")
frame_df = pd.read_csv(DATA_PATH + "framingham.csv")

# -----------------------------
# Basic Information
# -----------------------------
print("Heart Dataset Shape:", heart_df.shape)
print("Framingham Dataset Shape:", frame_df.shape)

# -----------------------------
# Dataset Info
# -----------------------------
heart_df.info()
frame_df.info()

# -----------------------------
# Missing Values
# -----------------------------
print("\nMissing Values (Heart Dataset):")
print(heart_df.isnull().sum())

print("\nMissing Values (Framingham Dataset):")
print(frame_df.isnull().sum())

# -----------------------------
# Target Distribution - Heart
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="target", data=heart_df)
plt.title("Heart Disease Distribution (Heart Dataset)")
plt.savefig(PLOT_PATH + "heart_target_distribution.png", bbox_inches="tight")
plt.show()

# -----------------------------
# Target Distribution - Framingham
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="TenYearCHD", data=frame_df)
plt.title("Heart Disease Distribution (Framingham Dataset)")
plt.savefig(PLOT_PATH + "framingham_target_distribution.png", bbox_inches="tight")
plt.show()

# -----------------------------
# Age vs Heart Disease
# -----------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x="target", y="age", data=heart_df)
plt.title("Age vs Heart Disease")
plt.savefig(PLOT_PATH + "age_vs_heart_disease.png", bbox_inches="tight")
plt.show()

# -----------------------------
# Cholesterol Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.histplot(frame_df["totChol"].dropna(), bins=30, kde=True)
plt.title("Total Cholesterol Distribution")
plt.savefig(PLOT_PATH + "cholesterol_distribution.png", bbox_inches="tight")
plt.show()

# -----------------------------
# Systolic Blood Pressure
# -----------------------------
plt.figure(figsize=(6,4))
sns.boxplot(y="sysBP", data=frame_df)
plt.title("Systolic Blood Pressure Distribution")
plt.savefig(PLOT_PATH + "systolic_bp_distribution.png", bbox_inches="tight")
plt.show()

# -----------------------------
# Correlation Heatmap - Heart
# -----------------------------
plt.figure(figsize=(10,8))
sns.heatmap(heart_df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap - Heart Dataset")
plt.savefig(PLOT_PATH + "heart_correlation_heatmap.png", bbox_inches="tight")
plt.show()

# -----------------------------
# Correlation Heatmap - Framingham
# -----------------------------
plt.figure(figsize=(10,8))
sns.heatmap(frame_df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap - Framingham Dataset")
plt.savefig(PLOT_PATH + "framingham_correlation_heatmap.png", bbox_inches="tight")
plt.show()

print("\n✅ EDA completed. All plots saved in outputs/plots/")
# -----------------------------
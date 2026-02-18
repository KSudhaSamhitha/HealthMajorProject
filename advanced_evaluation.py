import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

DATA_PATH = "processed_data/"
OUTPUT_PATH = "outputs/advanced_results/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

X = pd.read_csv(DATA_PATH + "X_scaled.csv")
y = pd.read_csv(DATA_PATH + "y.csv").values.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss",
                             scale_pos_weight=scale_pos_weight,
                             random_state=42)
}

def evaluate_model(model, name):
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)

    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.savefig(OUTPUT_PATH + f"roc_{name}.png")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(OUTPUT_PATH + f"cm_{name}.png")
    plt.close()

    return auc

results = {}

for name, model in models.items():
    auc = evaluate_model(model, name)
    results[name] = auc

print("\n=== 5-Fold Cross Validation (ROC-AUC) ===")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
    print(f"{name}: Mean AUC = {scores.mean():.4f}, Std = {scores.std():.4f}")

print("\n=== Hyperparameter Tuning: XGBoost ===")

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0]
}

xgb = XGBClassifier(eval_metric="logloss", random_state=42)

random_search = RandomizedSearchCV(
    xgb,
    param_distributions=param_grid,
    n_iter=10,
    scoring="roc_auc",
    cv=3,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best AUC:", random_search.best_score_)

best_xgb = random_search.best_estimator_

import joblib
joblib.dump(best_xgb, "outputs/results/Tuned_XGBoost.pkl")
print("✅ Tuned_XGBoost model saved successfully.")

evaluate_model(best_xgb, "Tuned_XGBoost")

clinical_features = [
    'age',
    'sex',
    'chest_pain_type',
    'resting_bp',
    'cholesterol',
    'fasting_blood_sugar',
    'resting_ecg',
    'max_heart_rate',
    'exercise_angina',
    'st_depression',
    'st_slope',
    'num_vessels',
    'thalassemia'
]


lifestyle_features = [
    'age',
    'sex',
    'education',
    'current_smoker',
    'cigs_per_day',
    'bp_meds',
    'prevalent_stroke',
    'prevalent_hypertension',
    'diabetes',
    'total_cholesterol',
    'sys_bp',
    'dia_bp',
    'bmi',
    'heart_rate',
    'glucose'
]


def compare_modalities(features, name):
    X_subset = X[features]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_subset, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_tr, y_tr)

    y_probs = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_probs)

    print(f"{name} AUC: {auc:.4f}")
    return auc

print("\n=== Multimodal Comparison ===")
auc_clinical = compare_modalities(clinical_features, "Clinical Only")
auc_lifestyle = compare_modalities(lifestyle_features, "Lifestyle Only")
auc_combined = compare_modalities(X.columns.tolist(), "Combined Fusion")


def evaluate_with_threshold(model, name, threshold=0.5):
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)

    print(f"\n{name} (Threshold = {threshold})")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    return rec

print("\n=== Threshold Tuning for Random Forest ===")

rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")

for t in [0.5, 0.45, 0.4, 0.35, 0.3]:
    evaluate_with_threshold(rf, "Random Forest", threshold=t)

def find_best_threshold_for_recall(model):
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]

    best_threshold = 0.5
    best_recall = 0

    for t in np.arange(0.2, 0.6, 0.02):
        y_pred = (y_probs >= t).astype(int)
        rec = recall_score(y_test, y_pred)

        if rec > best_recall:
            best_recall = rec
            best_threshold = t

    print("\nBest Threshold for Maximum Recall:")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Recall: {best_recall:.4f}")

    evaluate_with_threshold(model, "Random Forest (Optimized)", best_threshold)

find_best_threshold_for_recall(rf)

def tune_threshold(model, name):
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]

    print(f"\n=== Threshold Tuning for {name} ===")

    best_recall = 0
    best_threshold = 0.5

    for t in np.arange(0.2, 0.6, 0.02):
        y_pred = (y_probs >= t).astype(int)
        rec = recall_score(y_test, y_pred)

        if rec > best_recall:
            best_recall = rec
            best_threshold = t

        print(f"Threshold {t:.2f} → Recall: {rec:.4f}")

    print("\nBest Threshold for Maximum Recall:")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Recall: {best_recall:.4f}")

    # Final evaluation at best threshold
    y_pred = (y_probs >= best_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)

    print(f"\n{name} (Optimized Threshold = {best_threshold:.2f})")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

xgb_model = XGBClassifier(
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

tune_threshold(xgb_model, "XGBoost")

tune_threshold(best_xgb, "Tuned_XGBoost")

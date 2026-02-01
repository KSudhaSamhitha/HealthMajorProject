import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
THRESHOLD = 0.5

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "processed_data/"
OUTPUT_PATH = "outputs/results/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# -----------------------------
# Load Processed Data
# -----------------------------
X = pd.read_csv(DATA_PATH + "X_scaled.csv")
y = pd.read_csv(DATA_PATH + "y.csv").values.ravel()

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# -----------------------------
# Handle Class Imbalance for XGBoost
# -----------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
}


# -----------------------------
# Train & Evaluate
# -----------------------------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    # predict probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    # apply threshold
    y_pred = (y_probs >= THRESHOLD).astype(int)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    })

    # Save trained model
    import joblib
    joblib.dump(model, OUTPUT_PATH + f"{name.replace(' ', '_')}.pkl")

# -----------------------------
# Save Results
# -----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH + "model_comparison.csv", index=False)

print("\n✅ Model Training Completed")
print(results_df)

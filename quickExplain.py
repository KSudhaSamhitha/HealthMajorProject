# quickExplain.py
import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import argparse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def quick_explain(model_name, sample_idx=0):
    """
    Quick explanation function that can be called from command line
    """
    print("\n" + "="*70)
    print("QUICK EXPLAINABLE AI - HEART DISEASE PREDICTION")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    X = pd.read_csv("processed_data/X_scaled.csv")
    y = pd.read_csv("processed_data/y.csv").values.ravel()
    
    # Load model
    print(f"📂 Loading model: {model_name}")
    model_path = f"outputs/results/{model_name}.pkl"
    model = joblib.load(model_path)
    
    # Create train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Get sample
    sample = X_test.iloc[sample_idx:sample_idx+1]
    true_label = y_test[sample_idx]
    
    # Prediction
    pred = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]
    
    print("\n" + "="*70)
    print(f"📊 PREDICTION RESULTS")
    print("="*70)
    print(f"Sample Index:        {sample_idx}")
    print(f"True Label:          {'❤️ Heart Disease' if true_label==1 else '💚 No Heart Disease'}")
    print(f"Predicted:           {'❤️ Heart Disease' if pred==1 else '💚 No Heart Disease'}")
    print(f"Probability (HD):    {proba[1]:.3f}")
    print(f"Probability (No HD): {proba[0]:.3f}")
    
    # Feature values
    print("\n" + "="*70)
    print("📋 FEATURE VALUES FOR THIS SAMPLE")
    print("="*70)
    feature_values = {}
    for feature in X.columns:
        value = sample[feature].values[0]
        feature_values[feature] = value
        print(f"  {feature:20s}: {value:.3f}")
    
    # SHAP explanation
    print("\n" + "="*70)
    print("🔵 SHAP EXPLANATION")
    print("="*70)
    try:
        if any(x in model_name for x in ['XGBoost', 'Random_Forest', 'Gradient_Boosting']):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)
            
            if isinstance(shap_values, list):
                contributions = shap_values[1][0]
            else:
                contributions = shap_values[0]
            
            # Sort by absolute contribution
            feature_contrib = list(zip(X.columns, contributions))
            feature_contrib.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print("\nTop 5 features contributing to this prediction:")
            print("-" * 50)
            for feature, contrib in feature_contrib[:5]:
                direction = "⬆️ INCREASES risk" if contrib > 0 else "⬇️ DECREASES risk"
                impact = "strong" if abs(contrib) > 0.5 else "moderate" if abs(contrib) > 0.2 else "weak"
                print(f"  {feature:20s}: {contrib:+.3f} ({direction} - {impact} impact)")
            
            # Create SHAP waterfall plot
            plt.figure(figsize=(12, 6))
            if isinstance(shap_values, list):
                shap.waterfall_plot(shap.Explanation(values=shap_values[1][0], 
                                                    base_values=explainer.expected_value[1],
                                                    data=sample.values[0],
                                                    feature_names=X.columns.tolist()),
                                  show=False)
            else:
                shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                                    base_values=explainer.expected_value,
                                                    data=sample.values[0],
                                                    feature_names=X.columns.tolist()),
                                  show=False)
            plt.title(f'SHAP Waterfall - {model_name} - Sample {sample_idx}')
            plt.tight_layout()
            plt.savefig(f"shap_quick_{model_name}_sample{sample_idx}.png", 
                       bbox_inches='tight', dpi=100)
            plt.close()
            print(f"\n✅ SHAP waterfall plot saved: shap_quick_{model_name}_sample{sample_idx}.png")
            
        else:
            print("⚠️  SHAP explanation not available for this model type (KernelExplainer would be slow for quick explain)")
            
    except Exception as e:
        print(f"⚠️  SHAP explanation failed: {str(e)}")
    
    # LIME explanation
    print("\n" + "="*70)
    print("🟢 LIME EXPLANATION")
    print("="*70)
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X.columns.tolist(),
            class_names=['No Heart Disease', 'Heart Disease'],
            mode='classification',
            random_state=42
        )
        
        exp = explainer.explain_instance(
            data_row=sample.values[0],
            predict_fn=model.predict_proba,
            num_features=8
        )
        
        print("\nLocal interpretable explanation:")
        print("-" * 50)
        for feature, impact in exp.as_list(label=1):
            direction = "⬆️ INCREASES" if impact > 0 else "⬇️ DECREASES"
            print(f"  {feature:40s}: {impact:+.3f} ({direction} risk)")
        
        # Save LIME plot
        fig = plt.figure(figsize=(10, 6))
        exp_list = exp.as_list(label=1)
        if exp_list:
            features, impacts = zip(*exp_list)
            colors = ['red' if x < 0 else 'green' for x in impacts]
            
            plt.barh(features, impacts, color=colors)
            plt.xlabel('Impact on Prediction')
            plt.title(f'LIME Explanation - {model_name} - Sample {sample_idx}')
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            plt.tight_layout()
            plt.savefig(f"lime_quick_{model_name}_sample{sample_idx}.png", 
                       bbox_inches='tight', dpi=100)
            plt.close()
            print(f"\n✅ LIME plot saved: lime_quick_{model_name}_sample{sample_idx}.png")
            
    except Exception as e:
        print(f"⚠️  LIME explanation failed: {str(e)}")
    
    print("\n" + "="*70)
    print("✅ QUICK EXPLANATION COMPLETE")
    print("="*70)

def list_available_models():
    """List all available trained models"""
    import os
    models = [f.replace('.pkl', '') for f in os.listdir("outputs/results/") if f.endswith('.pkl')]
    return models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quick XAI explanation for heart disease prediction')
    parser.add_argument('--model', type=str, default='Logistic_Regression',
                       help='Model name (e.g., Logistic_Regression, XGBoost, Random_Forest, Gradient_Boosting)')
    parser.add_argument('--sample', type=int, default=0,
                       help='Sample index from test set to explain (0-100)')
    parser.add_argument('--list', action='store_true',
                       help='List available models')
    
    args = parser.parse_args()
    
    if args.list:
        print("\n📋 Available models:")
        for model in list_available_models():
            print(f"  - {model}")
    else:
        quick_explain(args.model, args.sample)
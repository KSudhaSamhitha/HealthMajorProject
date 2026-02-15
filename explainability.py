# explainability.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
import joblib
import os
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "processed_data/"
MODEL_PATH = "outputs/results/"
PLOT_PATH = "outputs/explainability_plots/"
os.makedirs(PLOT_PATH, exist_ok=True)

# -----------------------------
# Load Processed Data and Models
# -----------------------------
print("="*60)
print("EXPLAINABLE AI (XAI) FOR HEART DISEASE PREDICTION")
print("="*60)
print("\nLoading processed data and trained models...")

# Load data
X = pd.read_csv(DATA_PATH + "X_scaled.csv")
y = pd.read_csv(DATA_PATH + "y.csv").values.ravel()

print(f"✅ Data loaded: X shape {X.shape}, y shape {y.shape}")
print(f"✅ Number of features: {X.shape[1]}")
print(f"✅ Features: {list(X.columns)}")

# Load all trained models
models = {}
model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.pkl')]
for model_file in model_files:
    model_name = model_file.replace('.pkl', '').replace('_', ' ')
    models[model_name] = joblib.load(MODEL_PATH + model_file)

print(f"✅ Loaded models: {list(models.keys())}")

# Create train-test split for consistent explanations
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Train-test split: Train {X_train.shape}, Test {X_test.shape}")

# -----------------------------
# SHAP Explanations
# -----------------------------
def generate_shap_explanations(model, model_name, X_train_sample, X_test_sample):
    """
    Generate SHAP explanations for a trained model
    """
    print(f"\n{'='*50}")
    print(f"Generating SHAP explanations for {model_name}")
    print('='*50)
    
    try:
        # Create SHAP explainer based on model type
        if any(x in model_name for x in ['XGBoost', 'Gradient Boosting', 'Random Forest']):
            print(f"  Using TreeExplainer for {model_name}")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_sample)
            
                       # Handle binary classification output
            if isinstance(shap_values, list):
                shap_values_for_plot = shap_values[1]  # Use positive class
                expected_value = explainer.expected_value[1]
                print(f"  ✓ Using positive class SHAP values")
            else:
                shap_values_for_plot = shap_values
                expected_value = explainer.expected_value
            
            # Check if shap_values_for_plot has extra dimensions (like for Random Forest)
            if shap_values_for_plot.ndim > 2:
                print(f"  ⚠️ Reshaping SHAP values from {shap_values_for_plot.shape}")
                # If shape is (samples, features, classes), take the positive class
                if shap_values_for_plot.shape[-1] == 2:
                    shap_values_for_plot = shap_values_for_plot[..., 1]  # Take positive class
                else:
                    # Otherwise reshape to 2D
                    shap_values_for_plot = shap_values_for_plot.reshape(shap_values_for_plot.shape[0], -1)
                print(f"    Reshaped to: {shap_values_for_plot.shape}")
            
            # 1. Summary Plot
            plt.figure(figsize=(14, 10))
            shap.summary_plot(shap_values_for_plot, X_test_sample, 
                            feature_names=X.columns.tolist(), show=False)
            plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{PLOT_PATH}shap_summary_{model_name.replace(' ', '_')}.png", 
                       bbox_inches='tight', dpi=100)
            plt.close()
            print(f"  ✓ Summary plot saved")
            
            # 2. Bar Plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_for_plot, X_test_sample, 
                            feature_names=X.columns.tolist(), 
                            plot_type="bar", show=False)
            plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{PLOT_PATH}shap_bar_{model_name.replace(' ', '_')}.png", 
                       bbox_inches='tight', dpi=100)
            plt.close()
            print(f"  ✓ Bar plot saved")
            
            # 3. Waterfall Plot for first test sample - FIXED
            if len(X_test_sample) > 0:
                plt.figure(figsize=(14, 8))
                try:
                    # Check if shap_values_for_plot is 2D and has correct shape
                    if shap_values_for_plot.ndim == 2:
                        # Take first sample
                        shap_waterfall_values = shap_values_for_plot[0]
                    else:
                        shap_waterfall_values = shap_values_for_plot
                    
                    shap.waterfall_plot(shap.Explanation(values=shap_waterfall_values, 
                                                        base_values=expected_value,
                                                        data=X_test_sample.iloc[0].values,
                                                        feature_names=X.columns.tolist()),
                                      show=False, max_display=15)
                    plt.title(f'SHAP Waterfall - {model_name} (Sample 0)', fontsize=14)
                    plt.tight_layout()
                    plt.savefig(f"{PLOT_PATH}shap_waterfall_{model_name.replace(' ', '_')}.png", 
                               bbox_inches='tight', dpi=100)
                    plt.close()
                    print(f"  ✓ Waterfall plot saved")
                except Exception as e:
                    print(f"  ⚠️ Waterfall plot failed: {str(e)}")
            
                       # 4. Dependence Plots for top 5 features
            mean_shap = np.abs(shap_values_for_plot).mean(axis=0)
            top_features_idx = np.argsort(mean_shap)[-5:][::-1]
            
            # FIX: Convert to list to avoid pandas indexing issues
            # First convert to numpy array, then get indices, then get feature names
            top_features_idx_list = top_features_idx.tolist() if hasattr(top_features_idx, 'tolist') else list(top_features_idx)
            top_features = [X.columns[i] for i in top_features_idx_list]
            
            print(f"  Top 5 features: {top_features}")
            
            for feature in top_features:
                plt.figure(figsize=(12, 6))
                try:
                    shap.dependence_plot(feature, shap_values_for_plot, X_test_sample, 
                                       feature_names=X.columns.tolist(), show=False)
                    plt.title(f'SHAP Dependence - {feature} ({model_name})', fontsize=14)
                    plt.tight_layout()
                    plt.savefig(f"{PLOT_PATH}shap_dependence_{feature}_{model_name.replace(' ', '_')}.png", 
                               bbox_inches='tight', dpi=100)
                    plt.close()
                except Exception as e:
                    print(f"  ⚠️ Dependence plot for {feature} failed: {str(e)}")
            print(f"  ✓ Dependence plots saved for top features")
        else:  # Logistic Regression
            print(f"  Using KernelExplainer for {model_name} (this may take a few minutes)...")
            background = shap.sample(X_train_sample, min(50, len(X_train_sample)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_test_sample[:20], nsamples=100)
            
            # For KernelExplainer, shap_values is a list for binary classification
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values_for_plot = shap_values[1]  # Use positive class
                print(f"  ✓ Using positive class SHAP values")
            else:
                shap_values_for_plot = shap_values
            
            plt.figure(figsize=(14, 10))
            shap.summary_plot(shap_values_for_plot, X_test_sample[:20], 
                            feature_names=X.columns.tolist(), show=False)
            plt.title(f'SHAP Summary - {model_name}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{PLOT_PATH}shap_summary_{model_name.replace(' ', '_')}.png", 
                       bbox_inches='tight', dpi=100)
            plt.close()
            print(f"  ✓ Summary plot saved")
        
        print(f"✅ SHAP explanations completed for {model_name}")
        return shap_values, explainer
        
    except Exception as e:
        print(f"❌ Error generating SHAP for {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

# -----------------------------
# LIME Explanations - FIXED
# -----------------------------
def generate_lime_explanations(model, model_name, X_train, X_test, num_samples=3):
    """
    Generate LIME explanations for a trained model
    """
    print(f"\n{'='*50}")
    print(f"Generating LIME explanations for {model_name}")
    print('='*50)
    
    try:
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X.columns.tolist(),
            class_names=['No Heart Disease', 'Heart Disease'],
            mode='classification',
            discretize_continuous=True,
            random_state=42
        )
        
        # Generate explanations for multiple test samples
        for i in range(min(num_samples, len(X_test))):
            # Get prediction
            pred_proba = model.predict_proba(X_test.iloc[i:i+1])[0]
            pred_class = model.predict(X_test.iloc[i:i+1])[0]
            
            print(f"  Sample {i}: Predicted {'Heart Disease' if pred_class==1 else 'No Disease'} (prob: {pred_proba[1]:.3f})")
            
            # Generate explanation - FIX: Check which labels are available
            exp = explainer.explain_instance(
                data_row=X_test.iloc[i].values,
                predict_fn=model.predict_proba,
                num_features=min(10, len(X.columns)),
                top_labels=2  # Get both labels
            )
            
            # Check which labels are available in local_exp
            available_labels = list(exp.local_exp.keys())
            print(f"    Available labels: {available_labels}")
            
            # Use the first available label (usually 0 or 1)
            label_to_use = available_labels[0] if available_labels else 0
            
            # Save as HTML
            exp.save_to_file(f"{PLOT_PATH}lime_{model_name.replace(' ', '_')}_sample{i}.html")
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Feature importance plot - FIX: Use available label
            try:
                exp_list = exp.as_list(label=label_to_use)
                if exp_list:
                    features, impacts = zip(*exp_list)
                    colors = ['red' if x < 0 else 'green' for x in impacts]
                    
                    y_pos = np.arange(len(features))
                    ax1.barh(y_pos, impacts, color=colors)
                    ax1.set_yticks(y_pos)
                    ax1.set_yticklabels(features)
                    ax1.set_xlabel('Impact on Prediction')
                    ax1.set_title(f'Feature Contributions (for class {label_to_use})')
                    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            except Exception as e:
                print(f"    ⚠️ Could not create feature plot: {str(e)}")
                ax1.text(0.5, 0.5, 'Feature plot not available', 
                        ha='center', va='center', transform=ax1.transAxes)
            
            # Prediction probability plot
            ax2.bar(['No Heart Disease', 'Heart Disease'], pred_proba, color=['blue', 'red'])
            ax2.set_ylabel('Probability')
            ax2.set_title(f'Prediction: {"Heart Disease" if pred_class==1 else "No Heart Disease"} (Prob: {pred_proba[1]:.3f})')
            ax2.set_ylim([0, 1])
            for j, v in enumerate(pred_proba):
                ax2.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            plt.suptitle(f'LIME Explanation - {model_name} - Sample {i}', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{PLOT_PATH}lime_{model_name.replace(' ', '_')}_sample{i}.png", 
                       bbox_inches='tight', dpi=100)
            plt.close()
            
            print(f"  ✓ LIME explanation saved for sample {i}")
        
        print(f"✅ LIME explanations completed for {model_name}")
        return explainer
        
    except Exception as e:
        print(f"❌ Error generating LIME for {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# -----------------------------
# -----------------------------
# Cross-Model Comparison - FIXED
# -----------------------------
def compare_models_shap(models, X_test_sample):
    """
    Compare SHAP values across tree-based models
    """
    print(f"\n{'='*50}")
    print("Comparing feature importance across models")
    print('='*50)
    
    comparison_data = {}
    
    for model_name, model in models.items():
        try:
            if any(x in model_name for x in ['XGBoost', 'Random Forest', 'Gradient Boosting']):
                print(f"  Processing {model_name}...")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_sample)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    # For binary classification, take positive class
                    if len(shap_values) == 2:
                        shap_vals = shap_values[1]
                    else:
                        shap_vals = shap_values[0]
                else:
                    shap_vals = shap_values
                
                # Handle multi-dimensional SHAP values (like (50, 26, 2))
                if shap_vals.ndim == 3:
                    # Take the mean across the last dimension or select one class
                    shap_vals = shap_vals.mean(axis=-1)
                    print(f"    Reshaped from 3D to 2D: {shap_vals.shape}")
                
                # Calculate mean absolute SHAP values
                mean_shap = np.abs(shap_vals).mean(axis=0)
                
                # Ensure correct length (should be number of features)
                if len(mean_shap) > len(X.columns):
                    mean_shap = mean_shap[:len(X.columns)]
                elif len(mean_shap) < len(X.columns):
                    mean_shap = np.pad(mean_shap, (0, len(X.columns) - len(mean_shap)), 'constant')
                
                comparison_data[model_name] = mean_shap
                print(f"  ✓ Added {model_name} with shape {mean_shap.shape}")
        except Exception as e:
            print(f"  ✗ Could not add {model_name}: {str(e)}")
            continue
    
    if comparison_data:
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_data, index=X.columns)
        
        # Plot
        plt.figure(figsize=(16, 10))
        comparison_df.plot(kind='bar', ax=plt.gca())
        plt.title('Feature Importance Comparison Across Models', fontsize=16)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Mean |SHAP Value|', fontsize=12)
        plt.legend(title='Models', bbox_to_anchor=(1.05, 1))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{PLOT_PATH}model_comparison_shap.png", bbox_inches='tight', dpi=100)
        plt.close()
        
        print("✅ Model comparison plot saved")
        
        # Print top features
        print("\n📊 Top 10 features across models:")
        top_features = comparison_df.mean(axis=1).sort_values(ascending=False).head(10)
        for i, (feature, importance) in enumerate(top_features.items(), 1):
            print(f"  {i}. {feature}: {importance:.4f}")
        
        return comparison_df
    
    print("❌ No models could be compared")
    return None

# -----------------------------
# Individual Sample Explanation - FIXED
# -----------------------------
def explain_sample(model, model_name, sample_idx, X_train, X_test, y_test):
    """
    Provide detailed explanation for a specific sample
    """
    print(f"\n{'='*50}")
    print(f"DETAILED EXPLANATION FOR SAMPLE {sample_idx} - {model_name}")
    print('='*50)
    
    sample = X_test.iloc[sample_idx:sample_idx+1]
    true_label = y_test[sample_idx]
    pred = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]
    
    print(f"\nTrue Label: {'❤️ Heart Disease' if true_label==1 else '💚 No Heart Disease'}")
    print(f"Predicted: {'❤️ Heart Disease' if pred==1 else '💚 No Heart Disease'}")
    print(f"Probability: {proba[1]:.3f}")
    
    print("\n📋 Feature values for this sample:")
    feature_values = sample.iloc[0].to_dict()
    for feature, value in sorted(feature_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
        print(f"  {feature:25s}: {value:8.3f}")
    
    # Try SHAP
    try:
        if any(x in model_name for x in ['XGBoost', 'Random Forest', 'Gradient Boosting']):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)
            
            if isinstance(shap_values, list):
                contributions = shap_values[1][0]
            else:
                contributions = shap_values[0]
            
            print("\n🔵 SHAP Top Contributions:")
            feature_contrib = list(zip(X.columns, contributions))
            feature_contrib.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for feature, contrib in feature_contrib[:5]:
                direction = "⬆️ INCREASES" if contrib > 0 else "⬇️ DECREASES"
                print(f"  {feature:25s}: {contrib:+.3f} ({direction} risk)")
    except Exception as e:
        print(f"\n⚠️ SHAP explanation not available: {str(e)}")
    
    # Try LIME - FIXED
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X.columns.tolist(),
            class_names=['No Heart Disease', 'Heart Disease'],
            mode='classification'
        )
        
        exp = explainer.explain_instance(
            data_row=sample.values[0],
            predict_fn=model.predict_proba,
            num_features=8,
            top_labels=2
        )
        
        # Check available labels
        available_labels = list(exp.local_exp.keys())
        label_to_use = available_labels[0] if available_labels else 0
        
        print("\n🟢 LIME Top Contributions:")
        for feature, impact in exp.as_list(label=label_to_use)[:5]:
            direction = "⬆️ INCREASES" if impact > 0 else "⬇️ DECREASES"
            print(f"  {feature:40s}: {impact:+.3f} ({direction} risk)")
    except Exception as e:
        print(f"\n⚠️ LIME explanation not available: {str(e)}")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING EXPLAINABLE AI ANALYSIS")
    print("="*60)
    
    # Take samples for SHAP (to avoid memory issues)
    X_train_sample = X_train.sample(n=min(100, len(X_train)), random_state=42)
    X_test_sample = X_test.sample(n=min(50, len(X_test)), random_state=42)
    
    print(f"\nUsing {len(X_train_sample)} training samples and {len(X_test_sample)} test samples for SHAP")
    print(f"Number of features: {len(X.columns)}")
    
    # Generate explanations for each model
    for model_name, model in models.items():
        print(f"\n{'#'*60}")
        print(f"PROCESSING: {model_name}")
        print('#'*60)
        
        # SHAP explanations
        shap_values, explainer = generate_shap_explanations(
            model, model_name, X_train_sample, X_test_sample
        )
        
        # LIME explanations
        generate_lime_explanations(model, model_name, X_train, X_test_sample, num_samples=3)
    
    # Cross-model comparison
    compare_models_shap(models, X_test_sample)
    
    # Detailed explanation for first sample with each model
    print(f"\n{'#'*60}")
    print("DETAILED SAMPLE ANALYSIS")
    print('#'*60)
    
    for model_name, model in models.items():
        explain_sample(model, model_name, 0, X_train, X_test, y_test)
    
    print(f"\n{'='*60}")
    print(f"✅ EXPLAINABILITY ANALYSIS COMPLETE")
    print(f"✅ All plots saved in: {PLOT_PATH}")
    print('='*60)
    
    # List generated files
    print("\n📁 Generated files:")
    if os.path.exists(PLOT_PATH):
        for file in os.listdir(PLOT_PATH):
            print(f"  - {file}")
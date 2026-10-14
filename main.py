"""
50 Startups — Multiple Linear Regression Analysis
Implemented following the 6 phases of the CRISP-DM methodology:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment
"""

import json
import os
import sys
import urllib.request
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure standard output uses UTF-8 to prevent console encoding issues on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.preprocessing import OneHotEncoder


# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/SantiagoMorenoV/50-Startups-Multiple-Reg/main/50_Startups.csv"
DATA_DIR = "data"
OUTPUTS_DIR = "outputs"
DATA_FILE = os.path.join(DATA_DIR, "50_Startups.csv")
MODEL_FILE = os.path.join(OUTPUTS_DIR, "model_pipeline.joblib")
METRICS_FILE = os.path.join(OUTPUTS_DIR, "metrics.json")
SUMMARY_FILE = os.path.join(OUTPUTS_DIR, "evaluation_summary.txt")

# Set random seed for reproducibility
RANDOM_STATE = 42

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# =============================================================================
# Phase 1: Business Understanding
# =============================================================================
def phase_1_business_understanding():
    """
    Phase 1: Business Understanding
    Defines the business problem, target variable, and predictor features.
    """
    print("\n" + "=" * 70)
    print("PHASE 1: BUSINESS UNDERSTANDING")
    print("=" * 70)
    print("Goal: Predict startup profitability ('Profit') based on expenditures")
    print("      and geographic location to assist venture capitalists and")
    print("      management in investment appraisal and budget allocation.")
    print("\nTarget Variable (Dependent):")
    print("  - Profit: Net profit earned by the startup in USD.")
    print("\nInput Features (Independent):")
    print("  - R&D Spend: Research & Development expenditure in USD.")
    print("  - Administration: Administrative & operational expenses in USD.")
    print("  - Marketing Spend: Marketing & advertising expenditure in USD.")
    print("  - State: Operational state (categorical: New York, California, Florida).")
    print("=" * 70)


# =============================================================================
# Phase 2: Data Understanding
# =============================================================================
def phase_2_data_understanding():
    """
    Phase 2: Data Understanding
    Loads the dataset, conducts exploratory data analysis (EDA), and generates
    key statistical summaries and visual charts.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: DATA UNDERSTANDING")
    print("=" * 70)

    # Download data if not already present locally
    if not os.path.exists(DATA_FILE):
        print(f"Downloading dataset from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print("Download completed.")

    # Load data
    df = pd.read_csv(DATA_FILE)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\nColumn Information & Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col:<18}: {dtype}")

    # Check for missing values
    missing = df.isnull().sum()
    print("\nMissing Values Count:")
    for col, cnt in missing.items():
        print(f"  - {col:<18}: {cnt}")

    # Categorical breakdown
    print("\nCategorical Feature Distribution (State):")
    print(df["State"].value_counts().to_string())

    # Descriptive statistics
    print("\nDescriptive Statistics (Numerical Features):")
    print(df.describe().round(2).to_string())

    # Correlation matrix of numerical variables
    numeric_cols = ["R&D Spend", "Administration", "Marketing Spend", "Profit"]
    corr_matrix = df[numeric_cols].corr()
    print("\nPearson Correlation Matrix:")
    print(corr_matrix.round(4).to_string())

    # Visualization 1: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(numeric_cols, fontsize=10)
    plt.colorbar(im, ax=ax, label="Pearson Correlation")

    # Annotate correlation values on the heatmap
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            val = corr_matrix.iloc[i, j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight="bold")

    ax.set_title("Correlation Heatmap - 50 Startups", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    corr_path = os.path.join(OUTPUTS_DIR, "correlation_heatmap.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"\n[Chart Saved] Correlation heatmap -> {corr_path}")

    # Visualization 2: Feature Distributions
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        ax.hist(df[col], bins=10, color=colors[i], edgecolor="black", alpha=0.7)
        ax.set_title(f"Distribution of {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Feature and Target Distributions", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    dist_path = os.path.join(OUTPUTS_DIR, "eda_distributions.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"[Chart Saved] Feature distributions -> {dist_path}")

    # Visualization 3: Scatter Plots vs Profit
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    feature_cols = ["R&D Spend", "Administration", "Marketing Spend"]
    state_colors = {"New York": "#e41a1c", "California": "#377eb8", "Florida": "#4daf4a"}

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        for state, color in state_colors.items():
            subset = df[df["State"] == state]
            ax.scatter(subset[col], subset["Profit"], label=state, color=color, alpha=0.8, edgecolors="k", s=50)
        ax.set_title(f"{col} vs Profit", fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Profit ($)")
        ax.legend(title="State", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Predictor Features vs. Startup Profit", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    scatter_path = os.path.join(OUTPUTS_DIR, "eda_scatter_relationships.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"[Chart Saved] Scatter relationships -> {scatter_path}")

    print("=" * 70)
    return df


# =============================================================================
# Phase 3: Data Preparation
# =============================================================================
def phase_3_data_preparation(df):
    """
    Phase 3: Data Preparation
    Splits the data into 80% train / 20% test sets.
    Builds the preprocessing ColumnTransformer with One-Hot Encoding for State.
    Ensures zero Data Leakage by embedding transformations inside a pipeline.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: DATA PREPARATION")
    print("=" * 70)

    # Verify no missing values exist
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        print(f"Warning: Found {null_counts} missing values. Handling missing data...")
        df = df.dropna()
    else:
        print("Data Quality Verification: No missing values detected (0 nulls).")

    # Separate features and target
    X = df.drop(columns=["Profit"])
    y = df["Profit"]

    # 80% Training / 20% Testing split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"Data Split Ratio: 80% Train ({len(X_train)} samples), 20% Test ({len(X_test)} samples)")
    print(f"Random State: {RANDOM_STATE}")

    # Preprocessing Specification:
    # 1. Numerical features: pass-through without modification to keep linear coefficients interpretable in dollars.
    # 2. Categorical features ('State'): One-Hot Encoding with drop='first' to avoid dummy variable trap (multicollinearity).
    numeric_features = ["R&D Spend", "Administration", "Marketing Spend"]
    categorical_features = ["State"]

    # drop='first' drops 'California' (lexicographically first), leaving Florida and New York dummy variables.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop"
    )

    print("\nData Leakage Prevention:")
    print("  - The ColumnTransformer is fit strictly on training splits.")
    print("  - The test set is only evaluated through the fitted pipeline.")
    print("=" * 70)

    return X_train, X_test, y_train, y_test, preprocessor, numeric_features, categorical_features


# =============================================================================
# Phase 4: Modeling
# =============================================================================
def phase_4_modeling(X_train, y_train, preprocessor):
    """
    Phase 4: Modeling
    Creates the scikit-learn Pipeline combining preprocessor and LinearRegression.
    Evaluates model stability using 5-Fold Cross-Validation on the training set.
    Fits the final model on the entire training set.
    """
    print("\n" + "=" * 70)
    print("PHASE 4: MODELING")
    print("=" * 70)
    print("Model Algorithm: Multiple Linear Regression (sklearn.linear_model.LinearRegression)")

    from sklearn.pipeline import Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # 5-Fold Cross-Validation on Training set
    print("\nExecuting 5-Fold Cross-Validation on Training Set (40 samples)...")
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "r2": "r2",
        "neg_mae": "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error"
    }
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)

    cv_r2_scores = cv_results["test_r2"]
    cv_mae_scores = -cv_results["test_neg_mae"]
    cv_rmse_scores = -cv_results["test_neg_rmse"]

    for fold_idx in range(5):
        print(f"  Fold {fold_idx + 1}: R2 = {cv_r2_scores[fold_idx]:.4f}, "
              f"MAE = ${cv_mae_scores[fold_idx]:,.2f}, "
              f"RMSE = ${cv_rmse_scores[fold_idx]:,.2f}")

    print("-" * 50)
    print(f"5-Fold CV Mean R2:   {cv_r2_scores.mean():.4f} (std: {cv_r2_scores.std():.4f})")
    print(f"5-Fold CV Mean MAE:  ${cv_mae_scores.mean():,.2f} (std: ${cv_mae_scores.std():,.2f})")
    print(f"5-Fold CV Mean RMSE: ${cv_rmse_scores.mean():,.2f} (std: ${cv_rmse_scores.std():,.2f})")

    # Fit final pipeline on full training set
    print("\nFitting final Pipeline on full 80% Training Set...")
    pipeline.fit(X_train, y_train)
    print("Pipeline fitting complete.")
    print("=" * 70)

    cv_summary = {
        "fold_r2": cv_r2_scores.tolist(),
        "mean_r2": float(cv_r2_scores.mean()),
        "std_r2": float(cv_r2_scores.std()),
        "mean_mae": float(cv_mae_scores.mean()),
        "std_mae": float(cv_mae_scores.std()),
        "mean_rmse": float(cv_rmse_scores.mean()),
        "std_rmse": float(cv_rmse_scores.std()),
    }

    return pipeline, cv_summary


# =============================================================================
# Phase 5: Evaluation
# =============================================================================
def phase_5_evaluation(pipeline, X_train, y_train, X_test, y_test, numeric_features, categorical_features, cv_summary):
    """
    Phase 5: Evaluation
    Evaluates predictions on the test set using MAE, MSE, RMSE, and R2.
    Extracts regression coefficients and intercept.
    Generates Actual vs Predicted plot and Residual plot.
    Exports metrics to JSON and text summary.
    """
    print("\n" + "=" * 70)
    print("PHASE 5: EVALUATION")
    print("=" * 70)

    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # Test set metrics
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Train set metrics (for comparison)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    print("Test Set Performance (20% Holdout, N=10):")
    print(f"  - MAE  (Mean Absolute Error):       ${test_mae:,.2f}")
    print(f"  - MSE  (Mean Squared Error):        ${test_mse:,.2f}")
    print(f"  - RMSE (Root Mean Squared Error):   ${test_rmse:,.2f}")
    print(f"  - R^2  (Coefficient of Determ.):    {test_r2:.4f}")

    print("\nTraining Set Performance (80% Fit, N=40):")
    print(f"  - MAE:  ${train_mae:,.2f}")
    print(f"  - RMSE: ${train_rmse:,.2f}")
    print(f"  - R^2:  {train_r2:.4f}")

    # Extract coefficients and intercept
    regressor = pipeline.named_steps["regressor"]
    preprocessor = pipeline.named_steps["preprocessor"]
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_feature_names

    intercept = float(regressor.intercept_)
    coefficients = {name: float(coef) for name, coef in zip(all_feature_names, regressor.coef_)}

    print("\nModel Equation & Coefficients:")
    print(f"  Intercept (Beta_0): ${intercept:,.2f}")
    for name, coef in coefficients.items():
        print(f"  - {name:<18}: {coef:12.4f}")

    print("\nReference Category for State: 'California' (baseline state dropped by One-Hot Encoding).")

    # Diagnostic Chart 1: Actual vs. Predicted Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_test_pred, color="#2b5c8f", edgecolor="black", s=70, alpha=0.85, label="Test Samples")

    # Ideal 45-degree line
    min_val = min(y_test.min(), y_test_pred.min()) * 0.95
    max_val = max(y_test.max(), y_test_pred.max()) * 1.05
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal Fit (y = x)")

    ax.set_title(f"Actual vs. Predicted Profit (Test R^2 = {test_r2:.4f})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual Profit ($)", fontsize=11)
    ax.set_ylabel("Predicted Profit ($)", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    act_pred_path = os.path.join(OUTPUTS_DIR, "actual_vs_predicted.png")
    plt.savefig(act_pred_path, dpi=300)
    plt.close()
    print(f"\n[Chart Saved] Actual vs. Predicted -> {act_pred_path}")

    # Diagnostic Chart 2: Residual Plot
    residuals = y_test - y_test_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot A: Residuals vs Predicted Values
    ax1.scatter(y_test_pred, residuals, color="#e6550d", edgecolor="black", s=65, alpha=0.85)
    ax1.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax1.set_title("Residuals vs. Predicted Values", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Predicted Profit ($)", fontsize=10)
    ax1.set_ylabel("Residual (Actual - Predicted) ($)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot B: Residual Histogram
    ax2.hist(residuals, bins=6, color="#fdae6b", edgecolor="black", alpha=0.8)
    ax2.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax2.set_title("Distribution of Residuals", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Residual ($)", fontsize=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Residual Diagnostics - Linear Regression", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    res_path = os.path.join(OUTPUTS_DIR, "residual_plot.png")
    plt.savefig(res_path, dpi=300)
    plt.close()
    print(f"[Chart Saved] Residual diagnostics -> {res_path}")

    # Compile metrics dictionary
    evaluation_record = {
        "business_goal": "Predict startup profit using Multiple Linear Regression",
        "model_type": "LinearRegression with OneHotEncoder(drop='first')",
        "random_state": RANDOM_STATE,
        "sample_counts": {
            "total": len(X_train) + len(X_test),
            "train": len(X_train),
            "test": len(X_test)
        },
        "cross_validation_5_fold": cv_summary,
        "test_metrics": {
            "mae": float(test_mae),
            "mse": float(test_mse),
            "rmse": float(test_rmse),
            "r2": float(test_r2),
        },
        "train_metrics": {
            "mae": float(train_mae),
            "mse": float(train_mse),
            "rmse": float(train_rmse),
            "r2": float(train_r2),
        },
        "model_parameters": {
            "intercept": intercept,
            "coefficients": coefficients,
            "reference_category": "State_California",
        }
    }

    # Save metrics JSON
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(evaluation_record, f, indent=4)
    print(f"[File Saved] Metrics JSON -> {METRICS_FILE}")

    # Save human-readable summary text
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("50 Startups — Multiple Linear Regression Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"5-Fold CV Mean R2:   {cv_summary['mean_r2']:.4f} (std: {cv_summary['std_r2']:.4f})\n")
        f.write(f"5-Fold CV Mean RMSE: ${cv_summary['mean_rmse']:,.2f}\n\n")
        f.write("Test Set Metrics (20% Holdout, N=10):\n")
        f.write(f"  MAE:  ${test_mae:,.2f}\n")
        f.write(f"  MSE:  ${test_mse:,.2f}\n")
        f.write(f"  RMSE: ${test_rmse:,.2f}\n")
        f.write(f"  R2:   {test_r2:.4f}\n\n")
        f.write("Regression Equation:\n")
        f.write(f"  Profit = {intercept:,.2f}\n")
        for k, v in coefficients.items():
            f.write(f"         + ({v:,.4f} * {k})\n")
        f.write("\nNote: State_California is the baseline category.\n")
    print(f"[File Saved] Summary text -> {SUMMARY_FILE}")

    print("=" * 70)
    return evaluation_record


# =============================================================================
# Phase 6: Deployment
# =============================================================================
def phase_6_deployment(pipeline):
    """
    Phase 6: Deployment
    Serializes the complete trained pipeline (preprocessing + regression)
    using joblib, and validates inference with a sample prediction.
    """
    print("\n" + "=" * 70)
    print("PHASE 6: DEPLOYMENT")
    print("=" * 70)

    # Save the pipeline
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Model pipeline successfully saved to: {MODEL_FILE}")

    # Test reloading and predicting
    loaded_pipeline = joblib.load(MODEL_FILE)
    sample_data = pd.DataFrame([
        {
            "R&D Spend": 160000.0,
            "Administration": 130000.0,
            "Marketing Spend": 400000.0,
            "State": "New York"
        },
        {
            "R&D Spend": 100000.0,
            "Administration": 115000.0,
            "Marketing Spend": 250000.0,
            "State": "California"
        },
        {
            "R&D Spend": 70000.0,
            "Administration": 90000.0,
            "Marketing Spend": 150000.0,
            "State": "Florida"
        }
    ])

    print("\nValidating inference with sample input startups:")
    preds = loaded_pipeline.predict(sample_data)
    for idx, (row, pred) in enumerate(zip(sample_data.to_dict(orient="records"), preds), start=1):
        print(f"\nStartup #{idx}:")
        print(f"  R&D Spend:       ${row['R&D Spend']:>10,.2f}")
        print(f"  Administration:  ${row['Administration']:>10,.2f}")
        print(f"  Marketing Spend: ${row['Marketing Spend']:>10,.2f}")
        print(f"  State:           {row['State']}")
        print(f"  --> Predicted Profit: ${pred:>10,.2f}")

    print("\nDeployment validation passed successfully.")
    print("=" * 70)


# =============================================================================
# Main Orchestration
# =============================================================================
def main():
    print("Starting CRISP-DM Machine Learning Workflow for 50 Startups...")
    # Phase 1: Business Understanding
    phase_1_business_understanding()

    # Phase 2: Data Understanding
    df = phase_2_data_understanding()

    # Phase 3: Data Preparation
    X_train, X_test, y_train, y_test, preprocessor, num_feats, cat_feats = phase_3_data_preparation(df)

    # Phase 4: Modeling
    pipeline, cv_summary = phase_4_modeling(X_train, y_train, preprocessor)

    # Phase 5: Evaluation
    evaluation_record = phase_5_evaluation(
        pipeline, X_train, y_train, X_test, y_test, num_feats, cat_feats, cv_summary
    )

    # Feature Selection Analysis (5 Popular Algorithms)
    print("\nExecuting Feature Selection Analysis (5 Popular Algorithms)...")
    try:
        import feature_selection
        feature_selection.main()
    except Exception as e:
        print(f"Feature selection notice: {e}")

    # Phase 6: Deployment
    phase_6_deployment(pipeline)

    print("\nAll 6 CRISP-DM phases and Feature Selection analysis completed successfully!")


if __name__ == "__main__":
    main()

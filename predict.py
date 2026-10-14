"""
50 Startups — Model Prediction Script (CRISP-DM Phase 6: Deployment)
Loads the serialized Pipeline from 'outputs/model_pipeline.joblib'
and generates Profit predictions for new startup inputs.
"""

import argparse
import os
import sys
import joblib
import pandas as pd

# Ensure standard output uses UTF-8 to prevent console encoding issues on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


MODEL_PATH = os.path.join("outputs", "model_pipeline.joblib")


def load_model(model_path: str = MODEL_PATH):
    """
    Loads the trained scikit-learn Pipeline from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please run 'python main.py' first to train and export the pipeline."
        )
    return joblib.load(model_path)


def predict_profit(input_data, model_path: str = MODEL_PATH):
    """
    Generates Profit prediction(s) for the provided startup data.

    Parameters:
    -----------
    input_data : dict, list of dict, or pd.DataFrame
        Input data containing the required columns:
        ['R&D Spend', 'Administration', 'Marketing Spend', 'State']

    Returns:
    --------
    predictions : np.ndarray
        Predicted profit values in USD.
    """
    model = load_model(model_path)

    # Convert dictionary or list of dictionaries to DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError("input_data must be a dict, list of dicts, or pandas DataFrame.")

    # Validate required columns
    required_columns = ["R&D Spend", "Administration", "Marketing Spend", "State"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input is missing required column(s): {missing_cols}")

    return model.predict(df)


def run_demo():
    """
    Runs prediction demonstrations on sample startups across different states.
    """
    print("\n" + "=" * 65)
    print("50 Startups — Profit Prediction Demonstration")
    print("=" * 65)

    sample_startups = [
        {
            "name": "AlphaTech Innovations",
            "R&D Spend": 165349.20,
            "Administration": 136897.80,
            "Marketing Spend": 471784.10,
            "State": "New York"
        },
        {
            "name": "BayScale AI",
            "R&D Spend": 120000.00,
            "Administration": 110000.00,
            "Marketing Spend": 300000.00,
            "State": "California"
        },
        {
            "name": "SunCoast Logistics",
            "R&D Spend": 75000.00,
            "Administration": 95000.00,
            "Marketing Spend": 180000.00,
            "State": "Florida"
        },
        {
            "name": "SeedStage Lab",
            "R&D Spend": 20000.00,
            "Administration": 65000.00,
            "Marketing Spend": 50000.00,
            "State": "California"
        }
    ]

    df_samples = pd.DataFrame(sample_startups)
    predictions = predict_profit(df_samples[["R&D Spend", "Administration", "Marketing Spend", "State"]])

    for startup, pred in zip(sample_startups, predictions):
        print(f"\n[Startup: {startup['name']}]")
        print(f"  R&D Spend:        ${startup['R&D Spend']:>12,.2f}")
        print(f"  Administration:   ${startup['Administration']:>12,.2f}")
        print(f"  Marketing Spend:  ${startup['Marketing Spend']:>12,.2f}")
        print(f"  State:            {startup['State']}")
        print(f"  --> Predicted Profit: ${pred:>12,.2f}")

    print("\n" + "=" * 65)
    print("CLI Usage Example:")
    print("  python predict.py --rd 130000 --admin 120000 --marketing 320000 --state 'New York'")
    print("=" * 65 + "\n")


def parse_args():
    """
    Parses command-line arguments for single-sample inference.
    """
    parser = argparse.ArgumentParser(
        description="Predict startup profit using the trained Multiple Linear Regression pipeline."
    )
    parser.add_argument("--rd", type=float, help="R&D Spend in USD")
    parser.add_argument("--admin", type=float, help="Administration expenses in USD")
    parser.add_argument("--marketing", type=float, help="Marketing Spend in USD")
    parser.add_argument(
        "--state",
        type=str,
        choices=["New York", "California", "Florida"],
        help="Operating State (New York, California, Florida)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # If any parameter is passed, ensure all four are provided
    cli_params = [args.rd, args.admin, args.marketing, args.state]
    if any(p is not None for p in cli_params):
        if not all(p is not None for p in cli_params):
            print("Error: When providing CLI arguments, all four (--rd, --admin, --marketing, --state) must be specified.")
            sys.exit(1)

        input_data = {
            "R&D Spend": args.rd,
            "Administration": args.admin,
            "Marketing Spend": args.marketing,
            "State": args.state,
        }
        pred = predict_profit(input_data)[0]

        print("\nSingle Startup Profit Prediction:")
        print(f"  R&D Spend:        ${args.rd:,.2f}")
        print(f"  Administration:   ${args.admin:,.2f}")
        print(f"  Marketing Spend:  ${args.marketing:,.2f}")
        print(f"  State:            {args.state}")
        print(f"  --> Predicted Profit: ${pred:,.2f}\n")
    else:
        # Default to interactive/demo mode
        run_demo()


if __name__ == "__main__":
    main()

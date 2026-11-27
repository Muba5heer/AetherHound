
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = ROOT / "models" / "anomaly_model.pkl"
TEST_FILE = ROOT / "data" / "processed" / "test_dataset.csv"
OUTPUT_FILE = ROOT / "reports" / "evaluation_metrics.json"


def main():

    print("\nAETHERHOUND MODEL EVALUATION")
    print("-" * 40)

    # Check whether model exists
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model not found. Train the model first."
        )

    # Check whether test dataset exists
    if not TEST_FILE.exists():
        raise FileNotFoundError(
            "Test dataset not found. Run train_model.py first."
        )

    # Load trained model
    bundle = joblib.load(MODEL_FILE)

    model = bundle["model"]
    columns = bundle["feature_columns"]

    # Load test dataset WITHOUT changing column names
    df = pd.read_csv(TEST_FILE)

    print("\nTest dataset shape:", df.shape)

    # Check required columns
    missing_columns = [
        col for col in columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in test dataset: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    if "label" not in df.columns:
        raise ValueError("Label column is missing from test dataset.")

    # Select features using exact model column names
    X_test = df.loc[:, columns]
    y_test = df["label"].astype(int)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    # Display results
    print("\nMODEL EVALUATION RESULTS")
    print("-" * 40)

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1 Score:", round(f1, 3))

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test, predictions, labels=[0, 1]
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Anomalous"],
            zero_division=0
        )
    )

    # Save evaluation metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(OUTPUT_FILE, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nEvaluation completed successfully!")
    print("Report saved at:", OUTPUT_FILE)


if __name__ == "__main__":
    main()

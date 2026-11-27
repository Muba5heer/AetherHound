
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_FILE = ROOT / "reports" / "latest_prediction.json"
METRICS_FILE = ROOT / "reports" / "evaluation_metrics.json"
REPORT_FILE = ROOT / "reports" / "aetherhound_report.txt"


def main():

    print("\nAETHERHOUND REPORT GENERATOR")
    print("-" * 40)

    # Check prediction file
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            "Prediction file not found. Run main.py first."
        )

    # Read prediction results
    with open(PREDICTION_FILE, "r") as file:
        prediction_data = json.load(file)

    # Read evaluation metrics
    metrics = {}

    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r") as file:
            metrics = json.load(file)

    lines = []

    lines.append("=" * 55)
    lines.append("          AETHERHOUND DETECTION REPORT")
    lines.append("=" * 55)

    lines.append(
        f"\nAnalysis Date: {prediction_data['timestamp']}"
    )

    lines.append("\nAUDIO DETECTION RESULTS")
    lines.append("-" * 55)

    for result in prediction_data["results"]:
        lines.append(
            f"\nRecording: {result['recording_number']}"
        )
        lines.append(
            f"Result: {result['prediction']}"
        )
        lines.append(
            f"Anomaly Score: {result['anomaly_score']}%"
        )

    if metrics:
        lines.append("\nMODEL EVALUATION")
        lines.append("-" * 55)

        lines.append(
            f"Accuracy: {metrics['accuracy'] * 100:.2f}%"
        )
        lines.append(
            f"Precision: {metrics['precision']:.3f}"
        )
        lines.append(
            f"Recall: {metrics['recall']:.3f}"
        )
        lines.append(
            f"F1 Score: {metrics['f1_score']:.3f}"
        )

    lines.append("\nIMPORTANT LIMITATION")
    lines.append("-" * 55)
    lines.append(
        "Model predictions do not confirm a real covert channel."
    )
    lines.append(
        "Real-world validation using independent recordings is required."
    )

    lines.append("\n" + "=" * 55)
    lines.append("             END OF REPORT")
    lines.append("=" * 55)

    # Save report
    REPORT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\nReport generated successfully!")
    print("Saved at:", REPORT_FILE)


if __name__ == "__main__":
    main()


import os
import json
from datetime import datetime
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Load trained model
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "anomaly_model.pkl"
)

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
expected_columns = saved_data["feature_columns"]

# Load extracted audio features
FEATURE_PATH = os.path.join(
    BASE_DIR, "data", "processed", "audio_features.csv"
)

df = pd.read_csv(FEATURE_PATH)

# Match feature names with training dataset
aliases = {
    "zero_crossing_rate": "zcr",
    "spectral_centroid": "centroid",
    "spectral_bandwidth": "bandwidth",
    "spectral_rolloff": "rolloff"
}

def normalize(name):
    name = name.strip().lower().replace(" ", "_")
    return aliases.get(name, name)

# Normalize input feature names
df.columns = [normalize(c) for c in df.columns]

# Normalize expected training feature names
column_map = {
    normalize(c): c for c in expected_columns
}

# Check that all required features exist
missing = [
    c for c in expected_columns
    if normalize(c) not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required audio features: {missing}"
    )

# Arrange features in training order
X = df[
    [normalize(c) for c in expected_columns]
].copy()

X.columns = expected_columns

# Predict audio category
predictions = model.predict(X)
probabilities = model.predict_proba(X)

print("\nAETHERHOUND AUDIO ANALYSIS")
print("--------------------------")

anomaly_index = list(model.classes_).index(1)

# Store prediction results
results = []

for i, prediction in enumerate(predictions):

    if prediction == 0:
        result = "NORMAL AUDIO"
    else:
        result = "POTENTIAL ANOMALY DETECTED"

    score = probabilities[i][anomaly_index] * 100

    print(f"Result: {result}")
    print(f"Anomaly model score: {score:.2f}%")
    print("--------------------------")

    results.append({
        "recording_number": i + 1,
        "prediction": result,
        "anomaly_score": round(float(score), 2)
    })

print("Audio analysis completed!")

# Save prediction results as JSON
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    REPORT_DIR, "latest_prediction.json"
)

prediction_data = {
    "timestamp": datetime.now().astimezone().isoformat(),
    "results": results
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(prediction_data, file, indent=4)

print("\nPrediction results saved successfully!")
print("Saved at:", OUTPUT_FILE)



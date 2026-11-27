
import json
from datetime import datetime
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import sounddevice as sd
from scipy.io.wavfile import write


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = ROOT / "models" / "anomaly_model.pkl"

TEST_FOLDER = ROOT / "data" / "raw" / "test_recordings"

ANOMALY_FILE = (
    ROOT / "data" / "raw" / "simulated" /
    "anomalous" / "sample_0.wav"
)

CSV_REPORT = ROOT / "reports" / "test_results.csv"

JSON_REPORT = ROOT / "reports" / "test_results.json"

SAMPLE_RATE = 44100
DURATION = 5

TEST_FOLDER.mkdir(parents=True, exist_ok=True)
(ROOT / "reports").mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. RECORD AUDIO
# ==========================================

def record_audio(filename, test_name):

    input(
        f"\nPrepare for {test_name}. "
        "Press Enter to start recording..."
    )

    print(f"Recording {test_name} for {DURATION} seconds...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = audio.flatten()

    audio_int16 = np.clip(audio, -1, 1)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)

    file_path = TEST_FOLDER / filename

    write(str(file_path), SAMPLE_RATE, audio_int16)

    print("Recording saved:", file_path)

    return file_path


# ==========================================
# 3. EXTRACT 18 AUDIO FEATURES
# ==========================================

def extract_features(audio_path):

    audio, sr = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True
    )

    rms = np.mean(
        librosa.feature.rms(y=audio)
    )

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(y=audio)
    )

    centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=audio, sr=sr
        )
    )

    bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(
            y=audio, sr=sr
        )
    )

    rolloff = np.mean(
        librosa.feature.spectral_rolloff(
            y=audio, sr=sr
        )
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    features = {
        "RMS": rms,
        "ZCR": zcr,
        "Centroid": centroid,
        "Bandwidth": bandwidth,
        "Rolloff": rolloff
    }

    for i in range(13):
        features[f"MFCC_{i+1}"] = np.mean(mfcc[i])

    return features


# ==========================================
# 4. NORMALIZE FEATURE NAMES
# ==========================================

def normalize(name):

    name = str(name).strip().lower().replace(" ", "_")

    aliases = {
        "zero_crossing_rate": "zcr",
        "spectral_centroid": "centroid",
        "spectral_bandwidth": "bandwidth",
        "spectral_rolloff": "rolloff"
    }

    return aliases.get(name, name)


# ==========================================
# 5. MAIN TESTING FUNCTION
# ==========================================

def main():

    print("\nAETHERHOUND MODEL TESTING")
    print("=" * 50)

    # Check model
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Trained model not found! Train the model first."
        )

    # Load trained model
    bundle = joblib.load(MODEL_FILE)

    model = bundle["model"]

    expected_columns = bundle["feature_columns"]

    # Record normal room audio
    normal_file = record_audio(
        "normal_room.wav",
        "Normal room audio"
    )

    # Record quiet room audio
    quiet_file = record_audio(
        "quiet_room.wav",
        "Quiet room audio"
    )

    # Check simulated anomaly file
    if not ANOMALY_FILE.exists():
        raise FileNotFoundError(
            "Simulated anomaly sample not found. "
            "Run python audio/dataset_generation.py first."
        )

    print(
        "\nUsing simulated anomaly sample:",
        ANOMALY_FILE
    )

    # Prepare three test cases
    test_cases = [
        ("Normal Room", normal_file),
        ("Quiet Room", quiet_file),
        ("Simulated Anomaly", ANOMALY_FILE)
    ]

    results = []

    # ======================================
    # 6. TEST EACH AUDIO SAMPLE
    # ======================================

    for test_name, audio_file in test_cases:

        print(f"\nTesting: {test_name}")

        # Extract features
        features = extract_features(audio_file)

        df = pd.DataFrame([features])

        # IMPORTANT FIX:
        # Normalize DataFrame column names too
        df.columns = [
            normalize(column) for column in df.columns
        ]

        # Check missing features
        missing = [
            column for column in expected_columns
            if normalize(column) not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing features: {missing}"
            )

        # Arrange features in training order
        X = df[
            [normalize(column) for column in expected_columns]
        ].copy()

        X.columns = expected_columns

        # Predict audio category
        prediction = model.predict(X)[0]

        probabilities = model.predict_proba(X)

        anomaly_index = list(model.classes_).index(1)

        score = float(
            probabilities[0][anomaly_index] * 100
        )

        if prediction == 0:
            result = "NORMAL AUDIO"
        else:
            result = "POTENTIAL ANOMALY DETECTED"

        print("Result:", result)

        print(
            f"Anomaly model score: {score:.2f}%"
        )

        print("-" * 40)

        # Store results
        results.append({
            "test_name": test_name,
            "audio_file": str(audio_file.relative_to(ROOT)),
            "prediction": result,
            "anomaly_score": round(score, 2)
        })

    # ======================================
    # 7. SAVE TEST RESULTS
    # ======================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        CSV_REPORT,
        index=False
    )

    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "results": results
    }

    with open(
        JSON_REPORT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # ======================================
    # 8. DISPLAY FINAL COMPARISON
    # ======================================

    print("\n" + "=" * 65)

    print("AETHERHOUND TEST RESULTS")

    print("=" * 65)

    print(results_df.to_string(index=False))

    print("\nTesting completed successfully!")

    print("\nCSV report saved at:")
    print(CSV_REPORT)

    print("\nJSON report saved at:")
    print(JSON_REPORT)


if __name__ == "__main__":
    main()


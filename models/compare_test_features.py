
import librosa
import numpy as np
import pandas as pd
from pathlib import Path

files = {
    "Normal Room": "data/raw/test_recordings/normal_room.wav",
    "Quiet Room": "data/raw/test_recordings/quiet_room.wav",
    "Simulated Anomaly": "data/raw/simulated/anomalous/sample_0.wav"
}

def extract_features(path):
    audio, sr = librosa.load(path, sr=44100, mono=True)

    features = {
        "RMS": np.mean(librosa.feature.rms(y=audio)),
        "ZCR": np.mean(librosa.feature.zero_crossing_rate(y=audio)),
        "Centroid": np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
        "Bandwidth": np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)),
        "Rolloff": np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    }

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    for i in range(13):
        features[f"MFCC_{i+1}"] = np.mean(mfcc[i])

    return features

results = []

for name, path in files.items():
    if Path(path).exists():
        row = extract_features(path)
        row["Audio Type"] = name
        results.append(row)
    else:
        print("File missing:", path)

df = pd.DataFrame(results)

if not df.empty:
    df = df.set_index("Audio Type")
    print("\nAETHERHOUND TEST FEATURE COMPARISON")
    print("=" * 60)
    print(df.round(5).to_string())

    df.to_csv("reports/feature_comparison.csv")
    print("\nSaved: reports/feature_comparison.csv")
else:
    print("No audio files found.")


import os
import numpy as np
import pandas as pd
import librosa
from scipy.io.wavfile import write

# Dataset folders
normal_folder = "data/raw/simulated/normal"
anomaly_folder = "data/raw/simulated/anomalous"

os.makedirs(normal_folder, exist_ok=True)
os.makedirs(anomaly_folder, exist_ok=True)

SAMPLE_RATE = 44100
DURATION = 5
SAMPLES = SAMPLE_RATE * DURATION

dataset = []

def extract_features(audio):
    rms = np.mean(librosa.feature.rms(y=audio))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    centroid = np.mean(librosa.feature.spectral_centroid(
        y=audio, sr=SAMPLE_RATE
    ))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(
        y=audio, sr=SAMPLE_RATE
    ))
    rolloff = np.mean(librosa.feature.spectral_rolloff(
        y=audio, sr=SAMPLE_RATE
    ))

    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=13
    )
    mfcc_values = np.mean(mfcc, axis=1)

    features = {
        "RMS": rms,
        "ZCR": zcr,
        "Centroid": centroid,
        "Bandwidth": bandwidth,
        "Rolloff": rolloff
    }

    for i, value in enumerate(mfcc_values):
        features[f"MFCC_{i+1}"] = value

    return features


# Generate 50 normal and 50 simulated anomaly samples
for i in range(50):

    # Normal audio: low-level noise and hum
    t = np.arange(SAMPLES) / SAMPLE_RATE
    normal = (
        0.02 * np.sin(2 * np.pi * 120 * t)
        + 0.01 * np.random.randn(SAMPLES)
    )

    # Simulated anomaly: pulsed high-frequency tone
    pulse = (np.sin(2 * np.pi * 10 * t) > 0).astype(float)
    anomaly = (
        0.03 * np.sin(2 * np.pi * 2500 * t) * pulse
        + 0.01 * np.random.randn(SAMPLES)
    )

    for audio, folder, label in [
        (normal, normal_folder, 0),
        (anomaly, anomaly_folder, 1)
    ]:
        audio = np.clip(audio, -1, 1)
        filename = os.path.join(folder, f"sample_{i}.wav")

        write(filename, SAMPLE_RATE,
              (audio * 32767).astype(np.int16))

        features = extract_features(audio)
        features["label"] = label
        dataset.append(features)

# Save feature dataset
df = pd.DataFrame(dataset)
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/training_dataset.csv", index=False)

print("Dataset generated successfully!")
print("Total samples:", len(df))
print(df["label"].value_counts())
print("Saved: data/processed/training_dataset.csv")

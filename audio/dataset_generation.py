
import numpy as np
import pandas as pd
import librosa

from pathlib import Path
from scipy.io.wavfile import write

# Settings
SAMPLE_RATE = 22050
DURATION = 5
TOTAL_SAMPLES = 50

# Create folders
normal_folder = Path("data/raw/simulated/normal")
anomaly_folder = Path("data/raw/simulated/anomalous")

normal_folder.mkdir(parents=True, exist_ok=True)
anomaly_folder.mkdir(parents=True, exist_ok=True)


# Generate normal audio
def generate_normal_audio():

    t = np.linspace(
        0, DURATION,
        SAMPLE_RATE * DURATION,
        endpoint=False
    )

    # Background noise + ordinary low-frequency hum
    noise = np.random.normal(0, 0.03, len(t))
    hum = 0.15 * np.sin(2 * np.pi * 120 * t)

    audio = noise + hum

    return audio


# Generate simulated anomalous audio
def generate_anomalous_audio():

    t = np.linspace(
        0, DURATION,
        SAMPLE_RATE * DURATION,
        endpoint=False
    )

    # High-frequency carrier
    carrier = np.sin(2 * np.pi * 2500 * t)

    # Repeated on/off pulses
    pulse_rate = 10
    pulses = (np.sin(2 * np.pi * pulse_rate * t) > 0).astype(float)

    noise = np.random.normal(0, 0.02, len(t))

    audio = 0.6 * carrier * pulses + noise

    return audio


# Extract audio features
def extract_features(audio):

    rms = librosa.feature.rms(y=audio)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]

    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=SAMPLE_RATE
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=SAMPLE_RATE
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=SAMPLE_RATE
    )[0]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=13
    )

    features = {
        "RMS": np.mean(rms),
        "Zero_Crossing_Rate": np.mean(zcr),
        "Spectral_Centroid": np.mean(centroid),
        "Spectral_Bandwidth": np.mean(bandwidth),
        "Spectral_Rolloff": np.mean(rolloff)
    }

    for i in range(13):
        features[f"MFCC_{i+1}"] = np.mean(mfcc[i])

    return features


# Store dataset rows
dataset = []

print("Generating normal audio samples...")

for i in range(TOTAL_SAMPLES):

    audio = generate_normal_audio()

    filename = normal_folder / f"normal_{i+1}.wav"

    audio_int16 = np.int16(
        np.clip(audio, -1, 1) * 32767
    )

    write(filename, SAMPLE_RATE, audio_int16)

    features = extract_features(audio)
    features["label"] = "normal"

    dataset.append(features)


print("Generating anomalous audio samples...")

for i in range(TOTAL_SAMPLES):

    audio = generate_anomalous_audio()

    filename = anomaly_folder / f"anomaly_{i+1}.wav"

    audio_int16 = np.int16(
        np.clip(audio, -1, 1) * 32767
    )

    write(filename, SAMPLE_RATE, audio_int16)

    features = extract_features(audio)
    features["label"] = "anomalous"

    dataset.append(features)


# Save CSV
df = pd.DataFrame(dataset)

output_path = Path("data/processed/training_dataset.csv")
df.to_csv(output_path, index=False)

print("\nDataset generated successfully!")
print("Total samples:", len(df))
print("\nClass distribution:")
print(df["label"].value_counts())
print("\nDataset saved at:", output_path)


import librosa
import numpy as np
import pandas as pd
from pathlib import Path

# Input audio file
audio_path = "data/processed/filtered_audio.wav"

# Load audio
audio, sample_rate = librosa.load(
    audio_path,
    sr=None,
    mono=True
)

print("Audio loaded successfully!")
print("Sample rate:", sample_rate)

# Extract audio features
rms = librosa.feature.rms(y=audio)[0]
zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
centroid = librosa.feature.spectral_centroid(
    y=audio, sr=sample_rate
)[0]
bandwidth = librosa.feature.spectral_bandwidth(
    y=audio, sr=sample_rate
)[0]
rolloff = librosa.feature.spectral_rolloff(
    y=audio, sr=sample_rate
)[0]

# Extract MFCC features
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=13
)

# Calculate average values
features = {
    "RMS": np.mean(rms),
    "Zero_Crossing_Rate": np.mean(zcr),
    "Spectral_Centroid": np.mean(centroid),
    "Spectral_Bandwidth": np.mean(bandwidth),
    "Spectral_Rolloff": np.mean(rolloff)
}

# Add 13 MFCC values
for i in range(13):
    features[f"MFCC_{i+1}"] = np.mean(mfcc[i])

# Convert features into a table
df = pd.DataFrame([features])

# Save features
output_path = Path("data/processed/audio_features.csv")
df.to_csv(output_path, index=False)

print("\n--- EXTRACTED AUDIO FEATURES ---")
print(df.to_string(index=False))

print("\nFeatures saved successfully:", output_path)

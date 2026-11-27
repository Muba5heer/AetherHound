
import sounddevice as sd
import numpy as np
import pandas as pd
import librosa
from scipy.io.wavfile import write
from pathlib import Path

SAMPLE_RATE = 44100
DURATION = 5
TOTAL_SAMPLES = 10

OUTPUT_FOLDER = Path("data/raw/real_normal")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def extract_features(audio):
    features = {
        "RMS": np.mean(librosa.feature.rms(y=audio)),
        "ZCR": np.mean(librosa.feature.zero_crossing_rate(y=audio)),
        "Centroid": np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)
        ),
        "Bandwidth": np.mean(
            librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE)
        ),
        "Rolloff": np.mean(
            librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE)
        )
    }

    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=13
    )

    for i in range(13):
        features[f"MFCC_{i+1}"] = np.mean(mfcc[i])

    return features


def main():
    dataset = []

    print("\nAETHERHOUND NORMAL AUDIO COLLECTION")
    print("=" * 45)
    print("We will collect 10 normal audio recordings.")
    print("Record ordinary room sounds.")
    print("Avoid intentionally playing unusual test signals.")

    for i in range(TOTAL_SAMPLES):
        input(f"\nPress ENTER to record sample {i+1}/{TOTAL_SAMPLES}...")

        print("Recording for 5 seconds...")

        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )

        sd.wait()
        audio = audio.flatten()

        filename = OUTPUT_FOLDER / f"normal_{i+1}.wav"

        write(
            str(filename),
            SAMPLE_RATE,
            (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        )

        features = extract_features(audio)
        features["label"] = 0
        dataset.append(features)

        print("Saved:", filename)

    df = pd.DataFrame(dataset)

    output_file = OUTPUT_FOLDER / "real_normal_features.csv"
    df.to_csv(output_file, index=False)

    print("\nCOLLECTION COMPLETED!")
    print("Total recordings:", len(df))
    print("Feature dataset saved:", output_file)


if __name__ == "__main__":
    main()

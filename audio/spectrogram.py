
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
from pathlib import Path

audio_path = Path("data/processed/filtered_audio.wav")

if not audio_path.exists():
    print("Audio file not found. Run signal_processing.py first.")
    exit()

sample_rate, audio = wavfile.read(audio_path)

if audio.ndim == 2:
    audio = audio.mean(axis=1)

audio = audio.astype(float)

frequencies, times, power = spectrogram(
    audio,
    fs=sample_rate,
    nperseg=1024,
    noverlap=512
)

power_db = 10 * np.log10(power + 1e-10)

plt.figure(figsize=(12, 6))
plt.pcolormesh(times, frequencies, power_db, shading="auto")

plt.title("AetherHound Audio Spectrogram")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Power (dB)")

plt.tight_layout()

output_path = Path("data/processed/spectrogram.png")
plt.savefig(output_path)

print("Spectrogram saved successfully:", output_path)

plt.show()

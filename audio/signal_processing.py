
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt
from pathlib import Path


INPUT_FILE = Path("data/raw/sample_audio.wav")
OUTPUT_FOLDER = Path("data/processed")


def load_audio():
    try:
        sample_rate, audio = wavfile.read(INPUT_FILE)

        print("Audio loaded successfully!")
        print("Sample rate:", sample_rate)
        print("Audio shape:", audio.shape)

        return sample_rate, audio

    except Exception as e:
        print("Error loading audio:", e)
        return None, None


def normalize_audio(audio):
    if np.issubdtype(audio.dtype, np.integer):
        max_value = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32)

    return audio


def filter_noise(audio, sample_rate):
    # Keep frequencies between 100 Hz and 10,000 Hz
    nyquist = sample_rate / 2

    if nyquist <= 10000:
        raise ValueError("Sample rate is too low for this filter.")

    sos = butter(
        4,
        [100, 10000],
        btype="bandpass",
        fs=sample_rate,
        output="sos"
    )

    filtered = sosfiltfilt(sos, audio, axis=0)

    return filtered


def analyze_signal(audio, sample_rate):
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))

    frequencies = np.fft.rfftfreq(
        len(audio), d=1/sample_rate
    )

    fft_values = np.abs(np.fft.rfft(audio))

    # Ignore zero frequency
    dominant_index = np.argmax(fft_values[1:]) + 1
    dominant_frequency = frequencies[dominant_index]

    print("\n--- SIGNAL ANALYSIS ---")
    print("RMS Energy:", round(float(rms), 5))
    print("Peak Amplitude:", round(float(peak), 5))
    print("Dominant Frequency:",
          round(float(dominant_frequency), 2), "Hz")


def save_processed_audio(audio, sample_rate):
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_FOLDER / "filtered_audio.wav"

    audio = np.clip(audio, -1, 1)
    audio = (audio * 32767).astype(np.int16)

    wavfile.write(output_file, sample_rate, audio)

    print("\nProcessed audio saved:", output_file)


def main():
    sample_rate, audio = load_audio()

    if audio is None:
        return

    # Convert stereo to mono if necessary
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    audio = normalize_audio(audio)

    filtered_audio = filter_noise(audio, sample_rate)

    analyze_signal(filtered_audio, sample_rate)

    save_processed_audio(filtered_audio, sample_rate)

    print("\nSignal processing completed!")


if __name__ == "__main__":
    main()

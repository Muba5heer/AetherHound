
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pathlib import Path


# Audio settings
SAMPLE_RATE = 44100
DURATION = 10
CHANNELS = 1


def check_microphone():
    try:
        print("Available audio devices:")
        print(sd.query_devices())
        return True

    except Exception as e:
        print("Microphone error:", e)
        return False


def start_recording():
    try:
        print("\nRecording started...")
        print("Please keep your microphone active.")

        audio_data = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32"
        )

        sd.wait()

        print("Recording completed!")

        return audio_data

    except Exception as e:
        print("Recording error:", e)
        return None


def stop_recording():
    sd.stop()
    print("Recording stopped.")


def save_audio(audio_data):
    try:
        if audio_data is None:
            print("No audio data available.")
            return

        output_folder = Path("data/raw")
        output_folder.mkdir(parents=True, exist_ok=True)

        file_path = output_folder / "sample_audio.wav"

        # Convert audio into 16-bit format
        audio_data = np.clip(audio_data, -1, 1)
        audio_data = (audio_data * 32767).astype(np.int16)

        write(str(file_path), SAMPLE_RATE, audio_data)

        print(f"Audio saved successfully: {file_path}")

    except Exception as e:
        print("Saving error:", e)


if __name__ == "__main__":

    if check_microphone():
        recording = start_recording()
        save_audio(recording)


from audio.audio_sensor import check_microphone, start_recording, save_audio
import subprocess
import sys

def main():
    print("=" * 50)
    print("       AETHERHOUND DETECTOR")
    print(" Acoustic & RF Covert Channel Detection")
    print("=" * 50)

    print("\nSystem initialized successfully!")

    # Step 1: Check microphone
    print("\n[1] Checking microphone...")
    if not check_microphone():
        print("Microphone not available. Exiting.")
        return

    # Step 2: Record audio
    print("\n[2] Recording audio...")
    recording = start_recording()

    if recording is None:
        print("Recording failed. Exiting.")
        return

    # Step 3: Save audio
    print("\n[3] Saving audio...")
    save_audio(recording)

    # Step 4: Signal processing
    print("\n[4] Processing audio...")
    result = subprocess.run(
        [sys.executable, "audio/signal_processing.py"]
    )
    if result.returncode != 0:
        print("Signal processing failed. Exiting.")
        return

    # Step 5: Feature extraction
    print("\n[5] Extracting audio features...")
    result = subprocess.run(
        [sys.executable, "audio/feature_extraction.py"]
    )
    if result.returncode != 0:
        print("Feature extraction failed. Exiting.")
        return

    # Step 6: Anomaly prediction
    print("\n[6] Detecting anomalies...")
    result = subprocess.run(
        [sys.executable, "models/predict.py"]
    )
    if result.returncode != 0:
        print("Prediction failed.")

    print("\nAETHERHOUND ANALYSIS COMPLETED!")

if __name__ == "__main__":
    main()

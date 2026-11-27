# AetherHound

**Air-Gapped RF & Acoustic Covert Channel Detector** built on **GNU Radio**, **PyTorch**, **ZeroMQ**, and **Rich** for **Radioconda**.

AetherHound detects covert exfiltration where malware modulates CPU fan speeds or RAM/bus clocks to transmit Morse-code acoustic or electromagnetic pulses. It features a dual-mode GNU Radio signal pipeline so you can run a full end-to-end simulation on any laptop today and swap in physical RTL-SDR or ultrasonic microphone streams later.

---

## Architecture

| Module | Role |
| :--- | :--- |
| [`mock_malware.py`](mock_malware.py) | Covert transmitter simulator (`--mode ipc` 19.5 kHz ZMQ PUB Morse carrier + Gaussian noise; `--mode hardware` NumPy CPU-load Morse spikes) |
| [`dsp_pipeline.py`](dsp_pipeline.py) | Headless `gnuradio.gr.top_block` (ZMQ SUB / simulated noise / hardware adapter → Complex Bandpass FIR → `N_FFT=256` Blackman-Harris FFT → `Complex to Mag²` → log-dB normalization → non-blocking queue) |
| [`autoencoder.py`](autoencoder.py) | PyTorch 1D-Conv + Dense Autoencoder trained solely on baseline noise (`--record-baseline`, `--train`), computing threshold $\mu_{\text{loss}} + 3\sigma_{\text{loss}}$ and enforcing $K=5$ consecutive anomaly frames |
| [`cli_dashboard.py`](cli_dashboard.py) | 30+ FPS `rich` terminal UI (Unicode spectrum waterfall `▂▃▄▅▆▇█`, MSE vs. threshold meter, flashing red alert + `\a` bell) & forensic logger (`logs/detections.json`) |
| [`main.py`](main.py) | Unified CLI orchestrator (`setup`, `record-baseline`, `train`, `detect`, `malware`) |

---

## Prerequisites (Radioconda)

Open a **Radioconda Prompt** (or invoke `C:\Users\mubas\radioconda\python.exe` directly) and ensure `torch` and `rich` are installed alongside Radioconda's `gnuradio`, `numpy`, and `pyzmq`:

```powershell
pip install torch rich
```

---

## End-to-End Demo (2 Terminals)

### Step 1: Calibrate Baseline & Train Autoencoder (once)

```powershell
python main.py setup
```

Or run the individual module steps:

```powershell
python autoencoder.py --record-baseline --frames 1500
python autoencoder.py --train --epochs 35
```

This creates:
- `data/baseline.npy` — normalized baseline server-room noise spectrogram frames
- `models/aetherhound_ae.pth` — trained autoencoder weights and $\mu + 3\sigma$ threshold metadata

### Step 2: Terminal 1 — Start Detector & Live Dashboard

```powershell
python main.py detect
```

*(Or directly: `python cli_dashboard.py`)*

The dashboard launches at 30+ FPS in `[SECURE - BASELINE NORMAL]` state using the simulated baseline GNU Radio flowgraph while listening on `tcp://127.0.0.1:5555`.

### Step 3: Terminal 2 — Launch Mock Covert Malware

**IPC / Loopback Ultrasonic Carrier Mode (19.5 kHz Morse over ZeroMQ):**
```powershell
python mock_malware.py --mode ipc --message "AETHER EXFIL"
```

Watch Terminal 1 immediately switch source to `ZMQ-IPC`:
- The 19.5 kHz spike lights up the spectrum waterfall during Morse dots (`0.2s`) and dashes (`0.6s`).
- Reconstruction MSE crosses the $\mu + 3\sigma$ threshold for $K=5$ consecutive frames.
- Banner flashes red `[ALERT - COVERT CHANNEL DETECTED]`, sounds `\a`, and logs each confirmed exfiltration pulse to `logs/detections.json`.

**Hardware CPU-Load Mode (physical laptop CPU fan / EMI modulation):**
```powershell
python mock_malware.py --mode hardware --message "SOS"
```

---

## Swapping in Real SDR / Microphone Hardware

`dsp_pipeline.py` exposes `build_hardware_source()` supporting `"rtlsdr"` (`gnuradio-osmosdr`) and `"pyaudio"` (`gnuradio.audio`). Launch with:

```powershell
python main.py detect --source rtlsdr
# or
python main.py detect --source pyaudio
```

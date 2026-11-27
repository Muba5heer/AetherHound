#!/usr/bin/env python3
"""
tests/test_pipeline.py - End-to-end verification tests for AetherHound.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mock_malware
from autoencoder import AnomalyDetector, DEFAULT_MODEL_PATH
from cli_dashboard import ForensicLogger, run_dashboard
from dsp_pipeline import DSPConfig, SignalPipeline


def test_morse_schedule() -> None:
    sched = mock_malware.encode_morse_schedule("ET")
    marks = [(on, dur) for on, dur, _ in sched if on]
    assert marks == [(True, 0.2), (True, 0.6)], f"Unexpected Morse marks: {marks}"
    print("[PASS] test_morse_schedule")


def test_k_consecutive_threshold_logic() -> None:
    detector = AnomalyDetector(model_path=DEFAULT_MODEL_PATH, consecutive_k=5)
    # Normal baseline frame from data/baseline.npy should not trigger alert
    baseline = np.load(ROOT / "data" / "baseline.npy")
    false_alerts = 0
    for frame in baseline[:100]:
        res = detector.evaluate(frame)
        if res.alert_active:
            false_alerts += 1
    assert false_alerts == 0, f"Unexpected false alerts on baseline: {false_alerts}"

    # Synthetic transient spike for 4 frames (< K=5) must NOT trigger alert
    detector.reset()
    spike_frame = baseline[0].copy()
    spike_frame[232] = 0.99  # +19.5 kHz bin
    for i in range(4):
        res = detector.evaluate(spike_frame)
        assert res.is_frame_anomaly, f"Frame {i} should exceed threshold"
        assert not res.alert_active, f"Alert should not trigger at K={i+1} < 5"

    # 5th consecutive anomalous frame MUST trigger alert
    res5 = detector.evaluate(spike_frame)
    assert res5.alert_active and res5.consecutive_count == 5
    assert abs(res5.peak_freq_hz - 19500.0) < 1e-3, f"Peak freq={res5.peak_freq_hz}"
    print("[PASS] test_k_consecutive_threshold_logic")


def test_ipc_end_to_end_detection_and_logging() -> None:
    endpoint = "tcp://127.0.0.1:15555"
    detector = AnomalyDetector(model_path=DEFAULT_MODEL_PATH, consecutive_k=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "detections.json"
        logger = ForensicLogger(log_path=log_path)
        cfg = DSPConfig(mode="auto", ipc_endpoint=endpoint)

        with SignalPipeline(cfg) as pipe:
            # 1. Verify ambient SIM-BASELINE works immediately before IPC malware starts
            sim_tags = []
            for _ in range(15):
                item = pipe.read_frame(timeout=0.1)
                if item is not None:
                    frame, tag = item
                    sim_tags.append(tag)
                    res = detector.evaluate(frame)
                    logger.update(res, tag)
            assert any(t == "SIM-BASELINE" for t in sim_tags), f"Expected SIM-BASELINE, got {sim_tags}"

            # 2. Launch mock_malware in IPC noise-only mode -> should switch to ZMQ-IPC with 0 alerts
            stop_noise = threading.Event()
            noise_thread = threading.Thread(
                target=mock_malware.run_ipc_mode,
                kwargs={
                    "message": "E",
                    "stop_event": stop_noise,
                    "endpoint": endpoint,
                    "noise_only": True,
                },
                daemon=True,
            )
            noise_thread.start()
            ipc_tags = []
            noise_alerts = 0
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                item = pipe.read_frame(timeout=0.05)
                if item is not None:
                    frame, tag = item
                    if tag == "ZMQ-IPC":
                        ipc_tags.append(tag)
                        res = detector.evaluate(frame)
                        if res.alert_active:
                            noise_alerts += 1
            stop_noise.set()
            noise_thread.join(timeout=2.0)
            assert len(ipc_tags) >= 5, f"Expected ZMQ-IPC noise frames, got {len(ipc_tags)}"
            assert noise_alerts == 0, f"Unexpected alerts during noise-only IPC: {noise_alerts}"

            #Wait briefly for fallback holdoff
            time.sleep(0.25)

            # 3. Launch mock_malware in IPC Morse mode ("T" = single 0.6s dash at 19.5 kHz)
            stop_tx = threading.Event()
            tx_thread = threading.Thread(
                target=mock_malware.run_ipc_mode,
                kwargs={
                    "message": "T",
                    "stop_event": stop_tx,
                    "endpoint": endpoint,
                    "once": True,
                },
                daemon=True,
            )
            tx_thread.start()

            alert_seen = False
            peak_freqs = []
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                item = pipe.read_frame(timeout=0.05)
                if item is not None:
                    frame, tag = item
                    res = detector.evaluate(frame)
                    logger.update(res, tag)
                    if res.alert_active:
                        alert_seen = True
                        peak_freqs.append(res.peak_freq_hz)
                if not tx_thread.is_alive() and len(logger.events) > 0:
                    break
            stop_tx.set()
            tx_thread.join(timeout=2.0)

        logger.close()
        assert alert_seen, "Covert 19.5 kHz Morse pulse was not detected!"
        assert any(abs(f - 19500.0) < 250.0 for f in peak_freqs), (
            f"Expected peak near 19500 Hz, got {peak_freqs[:5]}"
        )
        assert log_path.exists(), "Forensic log file was not created"
        events = json.loads(log_path.read_text(encoding="utf-8"))
        assert len(events) >= 1, f"Expected >= 1 logged event, got {events}"
        ev = events[0]
        for key in ("timestamp", "peak_bin", "peak_freq_hz", "reconstruction_error", "duration_sec"):
            assert key in ev, f"Missing key {key} in {ev}"
        print(
            f"[PASS] test_ipc_end_to_end_detection_and_logging "
            f"(events={len(events)}, peak={ev['peak_freq_hz']} Hz, err={ev['reconstruction_error']})"
        )


def test_dashboard_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        rc = run_dashboard(
            model_path=ROOT / "models" / "aetherhound_ae.pth",
            log_path=Path(tmpdir) / "detections.json",
            source_mode="sim",
            duration=2.0,
            target_fps=35.0,
            enable_bell=False,
        )
        assert rc == 0
    print("[PASS] test_dashboard_smoke")


if __name__ == "__main__":
    test_morse_schedule()
    test_k_consecutive_threshold_logic()
    test_ipc_end_to_end_detection_and_logging()
    test_dashboard_smoke()
    print("\nALL AETHERHOUND TESTS PASSED!")

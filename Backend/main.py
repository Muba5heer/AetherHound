#!/usr/bin/env python3
"""
main.py - Unified Entry Point & Orchestrator for AetherHound.

Commands:
  detect           Launch the 30+ FPS AetherHound detector & dashboard
                   (auto-records baseline and trains model if not yet present).
  setup            Record baseline noise (data/baseline.npy) and train the
                   PyTorch autoencoder (models/aetherhound_ae.pth).
  record-baseline  Record baseline noise frames only.
  train            Train the autoencoder from data/baseline.npy only.
  malware          Run the mock covert transmitter simulator (IPC or hardware mode).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from autoencoder import (
    DEFAULT_BASELINE_PATH,
    DEFAULT_CONSECUTIVE_K,
    DEFAULT_MODEL_PATH,
    record_baseline,
    train_autoencoder,
)
from cli_dashboard import DEFAULT_LOG_PATH, run_dashboard
from dsp_pipeline import DEFAULT_IPC_ENDPOINT
import mock_malware


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aetherhound",
        description="AetherHound - Air-Gapped RF & Acoustic Covert Channel Detector",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # detect
    p_det = sub.add_parser(
        "detect",
        help="Start the live GNU Radio + PyTorch detector and Rich dashboard.",
    )
    p_det.add_argument(
        "--source",
        choices=("auto", "ipc", "sim", "rtlsdr", "pyaudio"),
        default="auto",
        help="Signal source mode (default: auto).",
    )
    p_det.add_argument("--endpoint", default=DEFAULT_IPC_ENDPOINT)
    p_det.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p_det.add_argument("--baseline-path", default=str(DEFAULT_BASELINE_PATH))
    p_det.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    p_det.add_argument("--k", type=int, default=DEFAULT_CONSECUTIVE_K)
    p_det.add_argument("--fps", type=float, default=34.0)
    p_det.add_argument("--duration", type=float, default=0.0)
    p_det.add_argument("--no-bell", action="store_true")
    p_det.add_argument(
        "--no-auto-setup",
        action="store_true",
        help="Fail instead of auto-recording/training when model checkpoint is missing.",
    )

    # setup
    p_setup = sub.add_parser(
        "setup",
        help="Record baseline noise and train the PyTorch autoencoder.",
    )
    p_setup.add_argument("--frames", type=int, default=1500)
    p_setup.add_argument("--epochs", type=int, default=35)
    p_setup.add_argument("--source", choices=("sim", "ipc", "auto"), default="sim")
    p_setup.add_argument("--endpoint", default=DEFAULT_IPC_ENDPOINT)
    p_setup.add_argument("--baseline-path", default=str(DEFAULT_BASELINE_PATH))
    p_setup.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))

    # record-baseline
    p_rec = sub.add_parser("record-baseline", help="Record baseline frames to data/baseline.npy.")
    p_rec.add_argument("--frames", type=int, default=1500)
    p_rec.add_argument("--source", choices=("sim", "ipc", "auto"), default="sim")
    p_rec.add_argument("--endpoint", default=DEFAULT_IPC_ENDPOINT)
    p_rec.add_argument("--baseline-path", default=str(DEFAULT_BASELINE_PATH))

    # train
    p_trn = sub.add_parser("train", help="Train autoencoder on data/baseline.npy.")
    p_trn.add_argument("--epochs", type=int, default=35)
    p_trn.add_argument("--baseline-path", default=str(DEFAULT_BASELINE_PATH))
    p_trn.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))

    # malware
    p_mal = sub.add_parser(
        "malware",
        help="Run mock_malware.py covert transmitter simulator.",
    )
    p_mal.add_argument("--mode", choices=("hardware", "ipc"), default="ipc")
    p_mal.add_argument("--message", default="AETHER EXFIL")
    p_mal.add_argument("--endpoint", default=DEFAULT_IPC_ENDPOINT)
    p_mal.add_argument("--carrier-freq", type=float, default=19_500.0)
    p_mal.add_argument("--amplitude", type=float, default=0.35)
    p_mal.add_argument("--noise-std", type=float, default=0.05)
    p_mal.add_argument("--once", action="store_true")
    p_mal.add_argument("--noise-only", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "setup":
        record_baseline(
            output_path=args.baseline_path,
            num_frames=args.frames,
            source_mode=args.source,
            ipc_endpoint=args.endpoint,
        )
        train_autoencoder(
            baseline_path=args.baseline_path,
            model_path=args.model_path,
            epochs=args.epochs,
        )
        return 0

    if args.command == "record-baseline":
        record_baseline(
            output_path=args.baseline_path,
            num_frames=args.frames,
            source_mode=args.source,
            ipc_endpoint=args.endpoint,
        )
        return 0

    if args.command == "train":
        train_autoencoder(
            baseline_path=args.baseline_path,
            model_path=args.model_path,
            epochs=args.epochs,
        )
        return 0

    if args.command == "detect":
        model_path = Path(args.model_path)
        baseline_path = Path(args.baseline_path)
        if not model_path.exists():
            if args.no_auto_setup:
                print(
                    f"[error] Model missing at {model_path}. Run 'python main.py setup' first.",
                    file=sys.stderr,
                )
                return 1
            print(
                "[main] No trained model found. Running automatic baseline calibration & training..."
            )
            if not baseline_path.exists():
                record_baseline(
                    output_path=baseline_path,
                    num_frames=1500,
                    source_mode="sim",
                    ipc_endpoint=args.endpoint,
                )
            train_autoencoder(
                baseline_path=baseline_path,
                model_path=model_path,
                epochs=35,
            )

        return run_dashboard(
            model_path=model_path,
            log_path=args.log_path,
            source_mode=args.source,
            ipc_endpoint=args.endpoint,
            consecutive_k=args.k,
            target_fps=args.fps,
            duration=args.duration,
            enable_bell=not args.no_bell,
        )

    if args.command == "malware":
        fwd_argv = [
            "--mode",
            args.mode,
            "--message",
            args.message,
            "--endpoint",
            args.endpoint,
            "--carrier-freq",
            str(args.carrier_freq),
            "--amplitude",
            str(args.amplitude),
            "--noise-std",
            str(args.noise_std),
        ]
        if args.once:
            fwd_argv.append("--once")
        if args.noise_only:
            fwd_argv.append("--noise-only")
        return mock_malware.main(fwd_argv)

    return 1


if __name__ == "__main__":
    sys.exit(main())

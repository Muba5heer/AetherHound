#!/usr/bin/env python3
"""
autoencoder.py - PyTorch Anomaly Detection Engine for AetherHound.

Trains a lightweight 1D-Conv + Dense Autoencoder solely on baseline ("normal")
server-room noise spectrogram frames, computes the statistical anomaly threshold
(mu_loss + 3 * sigma_loss), and exposes `AnomalyDetector` for real-time inference
with a K=5 consecutive-frame persistence filter.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dsp_pipeline import (
    DEFAULT_DB_MAX,
    DEFAULT_DB_MIN,
    DEFAULT_IPC_ENDPOINT,
    N_FFT,
    SAMPLE_RATE,
    DSPConfig,
    SignalPipeline,
)

DEFAULT_BASELINE_PATH = Path("data/baseline.npy")
DEFAULT_MODEL_PATH = Path("models/aetherhound_ae.pth")
DEFAULT_CONSECUTIVE_K = 5


class SpectrogramAutoencoder(nn.Module):
    """
    Lightweight 1D-Convolutional + Dense Autoencoder for N_FFT=256 spectrogram frames.
    Learns the normal spectral manifold of baseline server-room/RF background noise.
    """

    def __init__(self, n_fft: int = N_FFT, latent_dim: int = 32) -> None:
        super().__init__()
        if n_fft % 4 != 0:
            raise ValueError(f"n_fft must be divisible by 4, got {n_fft}")
        self.n_fft = n_fft
        self.latent_dim = latent_dim
        encoded_len = n_fft // 4
        flat_dim = 32 * encoded_len

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, latent_dim),
            nn.GELU(),
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.GELU(),
            nn.Unflatten(1, (32, encoded_len)),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder_conv(x.unsqueeze(1))
        z = self.encoder_fc(z)
        out = self.decoder_fc(z)
        out = self.decoder_conv(out)
        return out.squeeze(1)


@dataclass(frozen=True)
class DetectionResult:
    """Result returned by AnomalyDetector.evaluate() for a single FFT frame."""

    loss: float
    threshold: float
    is_frame_anomaly: bool
    alert_active: bool
    consecutive_count: int
    required_k: int
    peak_bin: int
    peak_freq_hz: float


class AnomalyDetector:
    """
    Real-time spectrogram anomaly detector wrapping the trained PyTorch autoencoder.
    Triggers an alert only when reconstruction MSE exceeds threshold for K consecutive frames.
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        consecutive_k: int = DEFAULT_CONSECUTIVE_K,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. "
                "Run '--record-baseline' and '--train' first."
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=True
        )

        self.n_fft = int(checkpoint["n_fft"])
        self.sample_rate = float(checkpoint["sample_rate"])
        self.latent_dim = int(checkpoint.get("latent_dim", 32))
        self.mu_loss = float(checkpoint["mu_loss"])
        self.sigma_loss = float(checkpoint["sigma_loss"])
        self.threshold = float(checkpoint["threshold"])
        self.db_min = float(checkpoint.get("db_min", DEFAULT_DB_MIN))
        self.db_max = float(checkpoint.get("db_max", DEFAULT_DB_MAX))

        self.model = SpectrogramAutoencoder(
            n_fft=self.n_fft, latent_dim=self.latent_dim
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.consecutive_k = max(1, int(consecutive_k))
        self._consecutive_hits = 0
        self.freq_axis_hz = np.fft.fftshift(
            np.fft.fftfreq(self.n_fft, d=1.0 / self.sample_rate)
        )

    def reset(self) -> None:
        """Reset consecutive anomaly frame counter."""
        self._consecutive_hits = 0

    def evaluate(self, frame: np.ndarray) -> DetectionResult:
        """
        Compute reconstruction MSE loss on one normalized FFT frame (shape: (n_fft,))
        and update the K-consecutive-frame alert state machine.
        """
        arr = np.asarray(frame, dtype=np.float32)
        if arr.shape != (self.n_fft,):
            raise ValueError(
                f"Expected frame shape ({self.n_fft},), got {arr.shape}"
            )

        with torch.inference_mode():
            tensor_in = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            recon = self.model(tensor_in)
            loss = float(F.mse_loss(recon, tensor_in).item())

        is_anomaly = loss > self.threshold
        if is_anomaly:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0

        alert_active = self._consecutive_hits >= self.consecutive_k
        peak_bin = int(np.argmax(arr))
        peak_freq_hz = float(self.freq_axis_hz[peak_bin])

        return DetectionResult(
            loss=loss,
            threshold=self.threshold,
            is_frame_anomaly=is_anomaly,
            alert_active=alert_active,
            consecutive_count=self._consecutive_hits,
            required_k=self.consecutive_k,
            peak_bin=peak_bin,
            peak_freq_hz=peak_freq_hz,
        )


def record_baseline(
    output_path: Path | str = DEFAULT_BASELINE_PATH,
    num_frames: int = 1500,
    source_mode: str = "sim",
    ipc_endpoint: str = DEFAULT_IPC_ENDPOINT,
    sample_rate: float = SAMPLE_RATE,
    n_fft: int = N_FFT,
) -> Path:
    """
    Capture `num_frames` normal baseline spectrogram frames from the GNU Radio
    DSP pipeline and persist them to `data/baseline.npy`.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()

    def _on_progress(done: int, total: int) -> None:
        pct = 100.0 * done / max(1, total)
        print(f"  [baseline] Captured {done}/{total} frames ({pct:.0f}%)", flush=True)

    cfg = DSPConfig(
        mode=source_mode,
        ipc_endpoint=ipc_endpoint,
        sample_rate=sample_rate,
        n_fft=n_fft,
    )
    print(
        f"[baseline] Recording {num_frames} normal noise frames "
        f"(source={source_mode}, N_FFT={n_fft})..."
    )
    with SignalPipeline(cfg) as pipe:
        frames = pipe.collect_frames(
            num_frames=num_frames,
            stop_event=stop_event,
            progress_cb=_on_progress,
        )

    if len(frames) == 0:
        raise RuntimeError("No baseline frames captured.")

    np.save(out, frames.astype(np.float32))
    print(
        f"[baseline] Saved shape={frames.shape} dtype={frames.dtype} -> {out}"
    )
    return out


def train_autoencoder(
    baseline_path: Path | str = DEFAULT_BASELINE_PATH,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    epochs: int = 35,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    latent_dim: int = 32,
    sample_rate: float = SAMPLE_RATE,
) -> dict:
    """
    Train the PyTorch autoencoder exclusively on baseline spectrogram frames,
    compute threshold = mu_loss + 3 * sigma_loss on validation frames, and save
    weights + threshold metadata to `models/aetherhound_ae.pth`.
    """
    b_path = Path(baseline_path)
    if not b_path.exists():
        raise FileNotFoundError(
            f"Baseline dataset not found at {b_path}. Run '--record-baseline' first."
        )

    data = np.load(b_path).astype(np.float32)
    if data.ndim != 2 or data.shape[0] < 20:
        raise ValueError(
            f"Expected 2D baseline array with >= 20 frames, got {data.shape}"
        )

    n_frames, n_fft = data.shape
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_frames)
    val_count = max(10, int(0.15 * n_frames))
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]

    train_tensor = torch.from_numpy(data[train_idx])
    val_tensor = torch.from_numpy(data[val_idx])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrogramAutoencoder(n_fft=n_fft, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=min(batch_size, len(train_tensor)),
        shuffle=True,
    )

    print(
        f"[train] Training on {len(train_tensor)} frames (val={len(val_tensor)}) "
        f"for {epochs} epochs on {device}..."
    )
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = F.mse_loss(recon, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * batch.size(0)
        epoch_loss /= len(train_tensor)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  [epoch {epoch:02d}/{epochs:02d}] train_mse={epoch_loss:.6f}")

    # Compute per-frame baseline validation MSE distribution (mu + 3*sigma)
    model.eval()
    with torch.inference_mode():
        val_in = val_tensor.to(device)
        val_recon = model(val_in)
        per_frame_mse = torch.mean((val_recon - val_in) ** 2, dim=1).cpu().numpy()

    mu_loss = float(np.mean(per_frame_mse))
    sigma_loss = float(np.std(per_frame_mse))
    threshold = float(mu_loss + 3.0 * sigma_loss)

    m_path = Path(model_path)
    m_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "n_fft": int(n_fft),
        "latent_dim": int(latent_dim),
        "sample_rate": float(sample_rate),
        "mu_loss": mu_loss,
        "sigma_loss": sigma_loss,
        "threshold": threshold,
        "db_min": float(DEFAULT_DB_MIN),
        "db_max": float(DEFAULT_DB_MAX),
        "epochs": int(epochs),
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    torch.save(payload, m_path)

    print(
        f"[train] Validation MSE: mu={mu_loss:.6f}, sigma={sigma_loss:.6f}, "
        f"threshold (mu+3*sigma)={threshold:.6f}"
    )
    print(f"[train] Saved model & threshold metadata -> {m_path}")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AetherHound PyTorch Autoencoder Anomaly Detection Engine"
    )
    parser.add_argument(
        "--record-baseline",
        action="store_true",
        help="Capture normal baseline spectrogram frames to data/baseline.npy.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train autoencoder on data/baseline.npy and save weights/threshold.",
    )
    parser.add_argument(
        "--baseline-path",
        default=str(DEFAULT_BASELINE_PATH),
        help=f"Baseline dataset path (default: {DEFAULT_BASELINE_PATH}).",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Model checkpoint path (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=1500,
        help="Number of baseline frames to capture (default: 1500).",
    )
    parser.add_argument(
        "--source",
        choices=("sim", "ipc", "auto"),
        default="sim",
        help="Signal source used during --record-baseline (default: sim).",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_IPC_ENDPOINT,
        help=f"ZMQ endpoint for IPC baseline recording (default: {DEFAULT_IPC_ENDPOINT}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
        help="Training epochs (default: 35).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.record_baseline and not args.train:
        parser.print_help()
        return 1

    signal.signal(signal.SIGINT, lambda *_: sys.exit(130))

    if args.record_baseline:
        record_baseline(
            output_path=args.baseline_path,
            num_frames=args.frames,
            source_mode=args.source,
            ipc_endpoint=args.endpoint,
        )

    if args.train:
        train_autoencoder(
            baseline_path=args.baseline_path,
            model_path=args.model_path,
            epochs=args.epochs,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

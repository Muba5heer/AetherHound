#!/usr/bin/env python3
"""
cli_dashboard.py - 30+ FPS Hacker-Style Rich Terminal Dashboard & Forensic Logger.

Displays:
  - System Status Banner: Green [SECURE - BASELINE NORMAL] vs. Flashing Red
    [ALERT - COVERT CHANNEL DETECTED] + terminal bell (\a) on alert trigger.
  - Real-time ASCII/Unicode Block-Character RF/Acoustic Spectrum Waterfall.
  - Live Autoencoder Reconstruction Loss vs. Threshold meter & telemetry.
  - Forensic Detection Event feed logged automatically to logs/detections.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from autoencoder import (
    DEFAULT_CONSECUTIVE_K,
    DEFAULT_MODEL_PATH,
    AnomalyDetector,
    DetectionResult,
)
from dsp_pipeline import (
    DEFAULT_IPC_ENDPOINT,
    DSPConfig,
    SignalPipeline,
)

DEFAULT_LOG_PATH = Path("logs/detections.json")

# Unicode block glyphs requested for spectrum waterfall rendering
WATERFALL_GLYPHS = np.array(list(" ▂▃▄▅▆▇█"))
WATERFALL_COLORS = (
    "grey23",
    "deep_sky_blue4",
    "dodger_blue2",
    "cyan2",
    "green3",
    "yellow1",
    "dark_orange",
    "bold bright_red",
)


class ForensicLogger:
    """
    Tracks confirmed covert-channel detection episodes and persists them to
    `logs/detections.json` as structured forensic records.
    """

    def __init__(self, log_path: Path | str = DEFAULT_LOG_PATH) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.events: List[Dict] = self._load_existing()
        self._active_episode: Optional[Dict] = None
        self._next_id = (
            max((int(e.get("event_id", 0)) for e in self.events), default=0) + 1
        )

    @property
    def active_episode(self) -> Optional[Dict]:
        """Return the currently ongoing alert episode, if any."""
        return self._active_episode

    def _load_existing(self) -> List[Dict]:
        if not self.log_path.exists():
            return []
        try:
            content = self.log_path.read_text(encoding="utf-8").strip()
            if not content:
                return []
            data = json.loads(content)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _flush(self) -> None:
        payload = json.dumps(self.events, indent=2)
        tmp_path = self.log_path.with_suffix(".tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        # Windows file indexers/editors can briefly hold a handle that blocks os.replace()
        for attempt in range(5):
            try:
                tmp_path.replace(self.log_path)
                return
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.015)
        # Fallback to direct overwrite if another process blocks atomic rename
        try:
            self.log_path.write_text(payload, encoding="utf-8")
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def update(self, result: DetectionResult, source_tag: str) -> bool:
        """
        Update active episode state from a DetectionResult.
        Returns True on the exact transition into a new confirmed alert episode.
        """
        now_mono = time.monotonic()
        now_iso = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
        rising_edge = False

        if result.alert_active:
            if self._active_episode is None:
                rising_edge = True
                self._active_episode = {
                    "event_id": self._next_id,
                    "timestamp": now_iso,
                    "start_mono": now_mono,
                    "peak_bin": result.peak_bin,
                    "peak_freq_hz": round(result.peak_freq_hz, 2),
                    "loss_sum": result.loss,
                    "max_loss": result.loss,
                    "threshold": result.threshold,
                    "frame_count": 1,
                    "source": source_tag,
                }
                self._next_id += 1
            else:
                ep = self._active_episode
                ep["frame_count"] += 1
                ep["loss_sum"] += result.loss
                if result.loss >= ep["max_loss"]:
                    ep["max_loss"] = result.loss
                    ep["peak_bin"] = result.peak_bin
                    ep["peak_freq_hz"] = round(result.peak_freq_hz, 2)
                ep["source"] = source_tag
        elif self._active_episode is not None:
            self._finalize_episode(now_mono, now_iso)

        return rising_edge

    def _finalize_episode(self, now_mono: float, now_iso: str) -> None:
        ep = self._active_episode
        if ep is None:
            return
        duration = max(0.001, now_mono - ep["start_mono"])
        record = {
            "event_id": ep["event_id"],
            "timestamp": ep["timestamp"],
            "end_timestamp": now_iso,
            "peak_bin": int(ep["peak_bin"]),
            "peak_freq_hz": float(ep["peak_freq_hz"]),
            "reconstruction_error": round(ep["loss_sum"] / ep["frame_count"], 6),
            "max_reconstruction_error": round(ep["max_loss"], 6),
            "threshold": round(ep["threshold"], 6),
            "duration_sec": round(duration, 3),
            "frame_count": int(ep["frame_count"]),
            "source": ep["source"],
        }
        self.events.append(record)
        self._active_episode = None
        self._flush()

    def close(self) -> None:
        if self._active_episode is not None:
            self._finalize_episode(
                time.monotonic(),
                dt.datetime.now().astimezone().isoformat(timespec="milliseconds"),
            )


class AetherHoundDashboard:
    """
    30+ FPS terminal UI rendering spectrum waterfall, autoencoder loss meter,
    status banner, and forensic event log.
    """

    def __init__(
        self,
        detector: AnomalyDetector,
        pipeline: SignalPipeline,
        logger: ForensicLogger,
        waterfall_rows: int = 16,
        target_fps: float = 34.0,
        enable_bell: bool = True,
    ) -> None:
        self.detector = detector
        self.pipeline = pipeline
        self.logger = logger
        self.waterfall_rows = waterfall_rows
        self.target_fps = target_fps
        self.enable_bell = enable_bell

        self.n_fft = detector.n_fft
        self.ring_buffer = np.zeros(
            (waterfall_rows, self.n_fft), dtype=np.float32
        )
        self.latest_result = DetectionResult(
            loss=0.0,
            threshold=detector.threshold,
            is_frame_anomaly=False,
            alert_active=False,
            consecutive_count=0,
            required_k=detector.consecutive_k,
            peak_bin=0,
            peak_freq_hz=0.0,
        )
        self.source_tag = "INIT"
        self.total_frames = 0
        self.render_fps = 0.0
        self.dsp_fps = 0.0
        self._loss_history: Deque[float] = deque(maxlen=40)
        self._start_time = time.monotonic()

    @staticmethod
    def _downsample_max_vectorized(
        matrix_2d: np.ndarray, target_cols: int
    ) -> np.ndarray:
        """
        Vectorized bin-wise max pooling across frequency axis so narrowband
        covert spikes are preserved at any terminal width.
        """
        rows, n_bins = matrix_2d.shape
        if target_cols >= n_bins:
            return matrix_2d
        # Group into target_cols contiguous buckets via index boundaries
        edges = np.linspace(0, n_bins, target_cols + 1, dtype=np.int32)
        out = np.empty((rows, target_cols), dtype=np.float32)
        for c in range(target_cols):
            lo, hi = int(edges[c]), int(edges[c + 1])
            if hi <= lo:
                hi = min(n_bins, lo + 1)
            out[:, c] = matrix_2d[:, lo:hi].max(axis=1)
        return out

    def _render_banner(self) -> Panel:
        now_t = time.monotonic()
        if self.latest_result.alert_active:
            flash_on = int(now_t * 4.0) % 2 == 0
            style = "bold white on red" if flash_on else "bold bright_red on grey11"
            banner_txt = Text(
                " [ALERT - COVERT CHANNEL DETECTED] ", style=style
            )
            border = "bright_red"
        else:
            banner_txt = Text(
                " [SECURE - BASELINE NORMAL] ", style="bold black on green3"
            )
            border = "green3"

        subtitle = Text(
            f"  |  SRC: {self.source_tag:<12s}  |  "
            f"UI: {self.render_fps:4.1f} FPS  |  "
            f"DSP: {self.dsp_fps:5.1f} frames/s  |  "
            f"PEAK: {self.latest_result.peak_freq_hz / 1000.0:+6.2f} kHz",
            style="bold cyan" if not self.latest_result.alert_active else "bold yellow",
        )
        combined = Text.assemble(banner_txt, subtitle)
        return Panel(
            Align.center(combined),
            title="[bold bright_cyan]AETHERHOUND // AIR-GAPPED RF & ACOUSTIC COVERT DETECTOR[/]",
            border_style=border,
            box=box.HEAVY,
        )

    def _render_waterfall(self, cols: int, max_rows: Optional[int] = None) -> Panel:
        cols = max(32, min(cols, 128))
        buf = self.ring_buffer
        if max_rows is not None and 1 <= max_rows < buf.shape[0]:
            buf = buf[-max_rows:]
        # Combine live top spectrum bar + scrolling waterfall history
        sampled = self._downsample_max_vectorized(buf, cols)
        levels = np.clip((sampled * 7.999).astype(np.int32), 0, 7)

        lines = Text()
        # Newest row at top
        for r_idx in range(levels.shape[0] - 1, -1, -1):
            row_lvls = levels[r_idx]
            row_chars = WATERFALL_GLYPHS[row_lvls]
            # Run-length encode colors along the row for fast Rich rendering
            start = 0
            cur_lvl = int(row_lvls[0])
            for c_idx in range(1, cols):
                lvl = int(row_lvls[c_idx])
                if lvl != cur_lvl:
                    lines.append(
                        "".join(row_chars[start:c_idx]),
                        style=WATERFALL_COLORS[cur_lvl],
                    )
                    start = c_idx
                    cur_lvl = lvl
            lines.append(
                "".join(row_chars[start:cols]), style=WATERFALL_COLORS[cur_lvl]
            )
            if r_idx > 0:
                lines.append("\n")

        # Frequency axis footer (-Fs/2 .. 0 .. +Fs/2)
        half_khz = self.detector.sample_rate / 2000.0
        left_lbl = f"-{half_khz:.0f} kHz"
        mid_lbl = "0 kHz"
        right_lbl = f"+{half_khz:.0f} kHz"
        pad_l = max(1, (cols - len(mid_lbl)) // 2 - len(left_lbl))
        pad_r = max(1, cols - len(left_lbl) - pad_l - len(mid_lbl) - len(right_lbl))
        axis_line = f"{left_lbl}{' ' * pad_l}{mid_lbl}{' ' * pad_r}{right_lbl}"
        lines.append(f"\n{axis_line}", style="dim white")

        return Panel(
            lines,
            title=f"[bold cyan]SPECTRUM WATERFALL (N_FFT={self.n_fft}, Bandpass 12-22.5 kHz)[/]",
            border_style="cyan",
            box=box.ROUNDED,
        )

    def _render_metrics(self) -> Panel:
        res = self.latest_result
        thresh = max(1e-9, res.threshold)
        scale_max = thresh * 2.5
        bar_width = 34
        fill_frac = min(1.0, res.loss / scale_max)
        thresh_pos = min(bar_width - 1, int((thresh / scale_max) * bar_width))
        filled = int(round(fill_frac * bar_width))

        meter = Text()
        for i in range(bar_width):
            if i == thresh_pos:
                meter.append("┃", style="bold white")
            elif i < filled:
                color = "bright_red" if res.loss > thresh else "green3"
                meter.append("█", style=color)
            else:
                meter.append("░", style="grey27")

        ratio = res.loss / thresh
        loss_style = "bold bright_red" if res.is_frame_anomaly else "bold green3"
        k_style = "bold bright_red" if res.alert_active else (
            "bold yellow" if res.consecutive_count > 0 else "cyan"
        )

        # Mini loss sparkline using Unicode blocks
        spark_chars = ""
        if self._loss_history:
            hist = np.array(list(self._loss_history), dtype=np.float32)
            norm_h = np.clip(hist / scale_max, 0.0, 1.0)
            idx_h = np.clip((norm_h * 7.999).astype(np.int32), 0, 7)
            spark_chars = "".join(WATERFALL_GLYPHS[idx_h])

        tbl = Table.grid(expand=True, padding=(0, 1))
        tbl.add_column(style="bold grey70", width=20)
        tbl.add_column()
        tbl.add_row("Loss Meter", meter)
        tbl.add_row(
            "MSE Loss",
            Text(f"{res.loss:.6f}  ({ratio:.2f}x threshold)", style=loss_style),
        )
        tbl.add_row(
            "Threshold (u+3s)",
            Text(
                f"{res.threshold:.6f}  (mu={self.detector.mu_loss:.6f}, "
                f"sigma={self.detector.sigma_loss:.6f})",
                style="white",
            ),
        )
        tbl.add_row(
            "Persistence (K)",
            Text(
                f"{res.consecutive_count} / {res.required_k} consecutive frames",
                style=k_style,
            ),
        )
        tbl.add_row(
            "Peak Bin / Freq",
            Text(
                f"Bin #{res.peak_bin:03d}  ({res.peak_freq_hz / 1000.0:+.3f} kHz)",
                style="bright_yellow" if res.alert_active else "cyan",
            ),
        )
        tbl.add_row(
            "Loss Trend",
            Text(spark_chars or "-", style="bright_red" if res.alert_active else "green3"),
        )
        tbl.add_row(
            "Processed Frames",
            Text(f"{self.total_frames:,} frames", style="dim white"),
        )

        return Panel(
            tbl,
            title="[bold magenta]AUTOENCODER RECONSTRUCTION & THRESHOLD METER[/]",
            border_style="magenta",
            box=box.ROUNDED,
        )

    def _render_events(self) -> Panel:
        tbl = Table(
            expand=True,
            box=box.SIMPLE_HEAD,
            show_edge=False,
            pad_edge=False,
        )
        tbl.add_column("ID", style="dim cyan", width=4)
        tbl.add_column("Time", style="white", width=12)
        tbl.add_column("Peak Freq", style="yellow", width=11)
        tbl.add_column("MSE Error", style="bright_red", width=10)
        tbl.add_column("Duration", style="cyan", width=8)
        tbl.add_column("Source", style="green3")

        if self.logger.active_episode is not None:
            ep = self.logger.active_episode
            dur = time.monotonic() - ep["start_mono"]
            ts_short = ep["timestamp"].split("T")[1][:12]
            tbl.add_row(
                f"#{ep['event_id']}",
                ts_short,
                f"{ep['peak_freq_hz']/1000.0:+.2f} kHz",
                f"{ep['max_loss']:.5f}",
                f"{dur:.2f}s*",
                f"{ep['source']} (LIVE)",
            )

        for ev in reversed(self.logger.events[-6:]):
            ts_short = str(ev["timestamp"]).split("T")[-1][:12]
            tbl.add_row(
                f"#{ev['event_id']}",
                ts_short,
                f"{ev['peak_freq_hz']/1000.0:+.2f} kHz",
                f"{ev['max_reconstruction_error']:.5f}",
                f"{ev['duration_sec']:.3f}s",
                str(ev["source"]),
            )

        if not self.logger.events and self.logger.active_episode is None:
            tbl.add_row("-", "--:--:--", "-", "-", "-", "No covert events logged")

        return Panel(
            tbl,
            title=f"[bold yellow]FORENSIC LOG ({self.logger.log_path})[/]",
            border_style="yellow",
            box=box.ROUNDED,
        )

    def build_layout(self, console_width: int, console_height: int = 34) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="banner", size=3),
            Layout(name="waterfall", ratio=3),
            Layout(name="bottom", size=12),
        )
        layout["bottom"].split_row(
            Layout(name="metrics", ratio=3),
            Layout(name="events", ratio=2),
        )
        wf_cols = max(40, console_width - 6)
        wf_rows = max(4, min(self.waterfall_rows, console_height - 18))
        layout["banner"].update(self._render_banner())
        layout["waterfall"].update(self._render_waterfall(wf_cols, max_rows=wf_rows))
        layout["metrics"].update(self._render_metrics())
        layout["events"].update(self._render_events())
        return layout

    def run(self, stop_event: threading.Event, duration: float = 0.0) -> None:
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        console = Console()
        render_interval = 1.0 / max(30.0, self.target_fps)
        next_render = time.monotonic()
        fps_window_start = time.monotonic()
        render_count = 0
        dsp_count = 0
        last_bell_time = 0.0

        with Live(
            self.build_layout(console.size.width, console.size.height),
            console=console,
            screen=console.is_terminal,
            auto_refresh=False,
            refresh_per_second=max(32, int(self.target_fps)),
        ) as live:
            while not stop_event.is_set():
                if duration > 0 and (time.monotonic() - self._start_time) >= duration:
                    break

                # Drain available frames since last cycle, tracking peak hold for display
                batch_peak_frame: Optional[np.ndarray] = None
                for _ in range(16):
                    item = self.pipeline.read_frame(timeout=0.005)
                    if item is None:
                        break
                    frame, tag = item
                    res = self.detector.evaluate(frame)
                    rising_edge = self.logger.update(res, tag)

                    self.latest_result = res
                    self.source_tag = tag
                    self.total_frames += 1
                    dsp_count += 1

                    if batch_peak_frame is None:
                        batch_peak_frame = frame.copy()
                    else:
                        np.maximum(batch_peak_frame, frame, out=batch_peak_frame)

                    if rising_edge and self.enable_bell:
                        now_b = time.monotonic()
                        if now_b - last_bell_time >= 0.8:
                            sys.stdout.write("\a")
                            sys.stdout.flush()
                            last_bell_time = now_b

                if batch_peak_frame is not None:
                    self.ring_buffer[:-1] = self.ring_buffer[1:]
                    self.ring_buffer[-1] = batch_peak_frame
                    self._loss_history.append(self.latest_result.loss)

                now = time.monotonic()
                if now >= next_render:
                    render_count += 1
                    elapsed_fps = now - fps_window_start
                    if elapsed_fps >= 0.5:
                        self.render_fps = render_count / elapsed_fps
                        self.dsp_fps = dsp_count / elapsed_fps
                        render_count = 0
                        dsp_count = 0
                        fps_window_start = now

                    live.update(
                        self.build_layout(console.size.width, console.size.height),
                        refresh=True,
                    )
                    next_render += render_interval
                    if next_render < now:
                        next_render = now + render_interval
                else:
                    time.sleep(min(0.005, max(0.0, next_render - now)))


def run_dashboard(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    log_path: Path | str = DEFAULT_LOG_PATH,
    source_mode: str = "auto",
    ipc_endpoint: str = DEFAULT_IPC_ENDPOINT,
    consecutive_k: int = DEFAULT_CONSECUTIVE_K,
    target_fps: float = 34.0,
    duration: float = 0.0,
    enable_bell: bool = True,
) -> int:
    """Start the DSP pipeline, autoencoder detector, forensic logger, and Rich UI."""
    detector = AnomalyDetector(
        model_path=model_path, consecutive_k=consecutive_k
    )
    logger = ForensicLogger(log_path=log_path)
    dsp_cfg = DSPConfig(
        mode=source_mode,
        ipc_endpoint=ipc_endpoint,
        n_fft=detector.n_fft,
        sample_rate=detector.sample_rate,
        db_min=detector.db_min,
        db_max=detector.db_max,
    )

    stop_event = threading.Event()

    def _handle_sigint(signum, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_sigint)

    try:
        with SignalPipeline(dsp_cfg) as pipe:
            dash = AetherHoundDashboard(
                detector=detector,
                pipeline=pipe,
                logger=logger,
                target_fps=target_fps,
                enable_bell=enable_bell,
            )
            dash.run(stop_event=stop_event, duration=duration)
    finally:
        logger.close()

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AetherHound Live Terminal Dashboard & Forensic Logger"
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help=f"Trained autoencoder path (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--log-path",
        default=str(DEFAULT_LOG_PATH),
        help=f"Forensic JSON log path (default: {DEFAULT_LOG_PATH}).",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "ipc", "sim", "rtlsdr", "pyaudio"),
        default="auto",
        help="Signal input mode (default: auto).",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_IPC_ENDPOINT,
        help=f"ZMQ SUB endpoint (default: {DEFAULT_IPC_ENDPOINT}).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_CONSECUTIVE_K,
        help=f"Consecutive anomaly frames required to alert (default: {DEFAULT_CONSECUTIVE_K}).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=34.0,
        help="Target dashboard refresh rate in FPS (default: 34.0).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Optional auto-exit duration in seconds (0 = run until Ctrl+C).",
    )
    parser.add_argument(
        "--no-bell",
        action="store_true",
        help="Disable terminal bell (\\a) on alert.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_dashboard(
        model_path=args.model_path,
        log_path=args.log_path,
        source_mode=args.source,
        ipc_endpoint=args.endpoint,
        consecutive_k=args.k,
        target_fps=args.fps,
        duration=args.duration,
        enable_bell=not args.no_bell,
    )


if __name__ == "__main__":
    sys.exit(main())

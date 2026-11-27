#!/usr/bin/env python3
"""
dsp_pipeline.py - Headless GNU Radio DSP Flowgraph for AetherHound.

Implements a dual-mode signal pipeline built on gnuradio.gr.top_block:
  - "ipc"  : Ingests live complex64 streams from a ZeroMQ SUB socket (tcp://127.0.0.1:5555).
  - "sim"  : Generates realistic Gaussian server-room baseline noise inside GNU Radio.
  - "auto" : Runs the live ZMQ SUB flowgraph alongside an ambient baseline GNU Radio
             flowgraph so the detector immediately displays baseline frames at 30+ FPS
             before mock_malware.py starts, and seamlessly switches to live ZMQ frames
             the moment the transmitter begins streaming.
  - Hardware adapters ("rtlsdr", "pyaudio") are cleanly abstracted so physical SDR or
    ultrasonic microphone streams can be swapped in with minimal changes.

Signal chain inside GNU Radio:
  Source -> Complex Bandpass FIR Filter -> Stream-to-Vector (N_FFT=256)
         -> Blackman-Harris Windowed FFT (shift=True) -> Complex to Mag^2
         -> SpectrogramSink (vectorized log-dB normalization -> non-blocking Queue / ZMQ PUB)
"""

from __future__ import annotations

import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import zmq
from gnuradio import analog, blocks, fft, filter, gr, zeromq

SAMPLE_RATE = 48_000.0
N_FFT = 256
DEFAULT_IPC_ENDPOINT = "tcp://127.0.0.1:5555"
DEFAULT_BANDPASS_LOW = 12_000.0
DEFAULT_BANDPASS_HIGH = 22_500.0
DEFAULT_TRANSITION_WIDTH = 1_000.0
DEFAULT_NOISE_STD = 0.05

# Fixed physical log-power dB bounds for consistent baseline vs. anomaly scaling
DEFAULT_DB_MIN = -45.0
DEFAULT_DB_MAX = 35.0


@dataclass(frozen=True)
class DSPConfig:
    """Configuration for the GNU Radio DSP flowgraph."""

    mode: str = "auto"  # "auto", "ipc", "sim", "rtlsdr", "pyaudio"
    ipc_endpoint: str = DEFAULT_IPC_ENDPOINT
    sample_rate: float = SAMPLE_RATE
    n_fft: int = N_FFT
    low_cutoff: float = DEFAULT_BANDPASS_LOW
    high_cutoff: float = DEFAULT_BANDPASS_HIGH
    transition_width: float = DEFAULT_TRANSITION_WIDTH
    noise_std: float = DEFAULT_NOISE_STD
    db_min: float = DEFAULT_DB_MIN
    db_max: float = DEFAULT_DB_MAX
    queue_maxsize: int = 128
    output_zmq_endpoint: Optional[str] = None
    # Hardware extension parameters (used when swapping in physical SDR / microphone)
    sdr_center_freq: float = 100e6
    sdr_gain: float = 30.0
    audio_device: Optional[str] = None


def build_hardware_source(config: DSPConfig):
    """
    Clean hardware abstraction hook for swapping in RTL-SDR or PyAudio streams.

    Returns a GNU Radio source block outputting `gr.sizeof_gr_complex` items.
    """
    if config.mode == "rtlsdr":
        try:
            import osmosdr  # type: ignore

            src = osmosdr.source(args="numchan=1")
            src.set_sample_rate(config.sample_rate)
            src.set_center_freq(config.sdr_center_freq, 0)
            src.set_gain(config.sdr_gain, 0)
            return src
        except ImportError as exc:
            raise RuntimeError(
                "RTL-SDR mode requested, but 'osmosdr' (gnuradio-osmosdr) or SDR hardware "
                "is unavailable. Use --source auto/ipc/sim for laptop testing."
            ) from exc

    if config.mode == "pyaudio":
        try:
            from gnuradio import audio  # type: ignore

            audio_src = audio.source(
                int(config.sample_rate), config.audio_device or "", True
            )
            f2c = blocks.float_to_complex(1)
            return (audio_src, f2c)
        except Exception as exc:
            raise RuntimeError(
                "Audio hardware mode requested, but GNU Radio audio source failed to open."
            ) from exc

    raise ValueError(f"Unsupported hardware source mode: {config.mode!r}")


class SpectrogramSink(gr.sync_block):
    """
    GNU Radio sink block that receives Mag^2 FFT vectors, applies vectorized
    log-dB normalization, and pushes frames onto a non-blocking queue.
    """

    def __init__(
        self,
        n_fft: int,
        frame_queue: queue.Queue,
        db_min: float = DEFAULT_DB_MIN,
        db_max: float = DEFAULT_DB_MAX,
        source_tag: str = "DSP",
        output_zmq_endpoint: Optional[str] = None,
    ) -> None:
        super().__init__(
            name="SpectrogramSink",
            in_sig=[(np.float32, n_fft)],
            out_sig=None,
        )
        self.n_fft = n_fft
        self.frame_queue = frame_queue
        self.db_min = float(db_min)
        self.db_inv_span = 1.0 / max(1e-6, float(db_max - db_min))
        self.source_tag = source_tag
        self.last_frame_time = 0.0

        self._zmq_ctx: Optional[zmq.Context] = None
        self._zmq_pub: Optional[zmq.Socket] = None
        if output_zmq_endpoint:
            self._zmq_ctx = zmq.Context()
            self._zmq_pub = self._zmq_ctx.socket(zmq.PUB)
            self._zmq_pub.setsockopt(zmq.SNDHWM, 64)
            self._zmq_pub.setsockopt(zmq.LINGER, 0)
            self._zmq_pub.bind(output_zmq_endpoint)

    def work(self, input_items, output_items) -> int:
        batch = input_items[0]
        n_vecs = len(batch)
        if n_vecs == 0:
            return 0

        # Vectorized log-power conversion and fixed-scale [0, 1] normalization
        db = 10.0 * np.log10(np.maximum(batch, 1e-12))
        norm_batch = np.clip((db - self.db_min) * self.db_inv_span, 0.0, 1.0).astype(
            np.float32
        )

        now = time.monotonic()
        self.last_frame_time = now

        for frame in norm_batch:
            frame_copy = frame.copy()
            if self._zmq_pub is not None:
                try:
                    self._zmq_pub.send(frame_copy.tobytes(), flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
            # Non-blocking enqueue: drop oldest frame if consumer lags
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.frame_queue.put_nowait((frame_copy, self.source_tag, now))
            except queue.Full:
                pass

        return n_vecs

    def close(self) -> None:
        if self._zmq_pub is not None:
            self._zmq_pub.close(linger=0)
            self._zmq_pub = None
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
            self._zmq_ctx = None


class AetherHoundFlowgraph(gr.top_block):
    """
    Headless GNU Radio top_block performing bandpass filtering, FFT, and Mag^2.
    """

    def __init__(
        self,
        config: DSPConfig,
        source_kind: str,
        frame_queue: queue.Queue,
        seed: int = 42,
    ) -> None:
        super().__init__(f"AetherHound_{source_kind}")
        self.config = config
        self.source_kind = source_kind

        # 1. Signal Source
        extra_tail = None
        if source_kind == "ipc":
            self.source = zeromq.sub_source(
                gr.sizeof_gr_complex,
                1,
                config.ipc_endpoint,
                timeout=100,
                pass_tags=False,
                hwm=-1,
                key="",
            )
            source_tag = "ZMQ-IPC"
        elif source_kind == "sim":
            # Complex Gaussian noise matching mock_malware baseline power (std / sqrt(2) per I/Q)
            self.noise_src = analog.noise_source_c(
                analog.GR_GAUSSIAN, float(config.noise_std), seed
            )
            self.throttle = blocks.throttle(
                gr.sizeof_gr_complex, float(config.sample_rate), True
            )
            self.connect(self.noise_src, self.throttle)
            self.source = self.throttle
            source_tag = "SIM-BASELINE"
        elif source_kind in ("rtlsdr", "pyaudio"):
            hw = build_hardware_source(config)
            if isinstance(hw, tuple):
                self.connect(hw[0], hw[1])
                self.source = hw[1]
                extra_tail = hw[0]
            else:
                self.source = hw
            source_tag = source_kind.upper()
        else:
            raise ValueError(f"Unknown flowgraph source_kind: {source_kind!r}")
        self._extra_hw = extra_tail

        # 2. Complex Bandpass FIR Filter isolating covert RF/ultrasonic band
        taps = filter.firdes.complex_band_pass(
            1.0,
            float(config.sample_rate),
            float(config.low_cutoff),
            float(config.high_cutoff),
            float(config.transition_width),
        )
        self.bpf = filter.fir_filter_ccc(1, taps)

        # 3. Stream to Vector (N_FFT = 256)
        self.s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, config.n_fft)

        # 4. Windowed FFT (shift=True orders frequencies -Fs/2 .. +Fs/2)
        win = fft.window.blackmanharris(config.n_fft)
        self.fft_blk = fft.fft_vcc(config.n_fft, True, win, True, 1)

        # 5. Complex to Mag^2
        self.c2m = blocks.complex_to_mag_squared(config.n_fft)

        # 6. Spectrogram Sink
        self.sink = SpectrogramSink(
            n_fft=config.n_fft,
            frame_queue=frame_queue,
            db_min=config.db_min,
            db_max=config.db_max,
            source_tag=source_tag,
            output_zmq_endpoint=config.output_zmq_endpoint,
        )

        self.connect(self.source, self.bpf, self.s2v, self.fft_blk, self.c2m, self.sink)

    def shutdown(self) -> None:
        """Stop and wait cleanly without hanging on ZMQ or GNU Radio threads."""
        try:
            self.stop()
        except Exception:
            pass
        wait_thread = threading.Thread(target=self.wait, daemon=True)
        wait_thread.start()
        wait_thread.join(timeout=1.5)
        self.sink.close()


class SignalPipeline:
    """
    High-level dual-mode signal acquisition controller.

    Manages the headless GNU Radio flowgraph(s) and exposes non-blocking frame
    consumption for the autoencoder and live CLI dashboard.
    """

    def __init__(self, config: Optional[DSPConfig] = None) -> None:
        self.config = config or DSPConfig()
        self.ipc_queue: queue.Queue = queue.Queue(maxsize=self.config.queue_maxsize)
        self.sim_queue: queue.Queue = queue.Queue(maxsize=self.config.queue_maxsize)

        self._ipc_tb: Optional[AetherHoundFlowgraph] = None
        self._sim_tb: Optional[AetherHoundFlowgraph] = None
        self._last_ipc_rx: float = 0.0
        self._started = False

    @property
    def freq_axis_hz(self) -> np.ndarray:
        """Return frequency bin centers in Hz matching fft_vcc(..., shift=True)."""
        return np.fft.fftshift(
            np.fft.fftfreq(self.config.n_fft, d=1.0 / self.config.sample_rate)
        )

    def start(self) -> None:
        if self._started:
            return
        mode = self.config.mode.lower()
        if mode == "auto":
            self._ipc_tb = AetherHoundFlowgraph(self.config, "ipc", self.ipc_queue)
            self._sim_tb = AetherHoundFlowgraph(self.config, "sim", self.sim_queue)
            self._ipc_tb.start()
            self._sim_tb.start()
        elif mode == "ipc":
            self._ipc_tb = AetherHoundFlowgraph(self.config, "ipc", self.ipc_queue)
            self._ipc_tb.start()
        elif mode == "sim":
            self._sim_tb = AetherHoundFlowgraph(self.config, "sim", self.sim_queue)
            self._sim_tb.start()
        elif mode in ("rtlsdr", "pyaudio"):
            self._ipc_tb = AetherHoundFlowgraph(self.config, mode, self.ipc_queue)
            self._ipc_tb.start()
        else:
            raise ValueError(f"Unsupported pipeline mode: {self.config.mode!r}")
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        if self._ipc_tb is not None:
            self._ipc_tb.shutdown()
            self._ipc_tb = None
        if self._sim_tb is not None:
            self._sim_tb.shutdown()
            self._sim_tb = None
        self._started = False

    def __enter__(self) -> "SignalPipeline":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def read_frame(self, timeout: float = 0.05) -> Optional[Tuple[np.ndarray, str]]:
        """
        Retrieve the next normalized spectrogram frame (shape: (N_FFT,), float32)
        along with the active source tag ("ZMQ-IPC", "SIM-BASELINE", etc.).
        """
        # Prefer live IPC / hardware stream whenever packets are arriving
        try:
            frame, tag, ts = self.ipc_queue.get_nowait()
            self._last_ipc_rx = ts
            # Drain ambient sim queue so it doesn't lag behind
            while not self.sim_queue.empty():
                try:
                    self.sim_queue.get_nowait()
                except queue.Empty:
                    break
            return frame, tag
        except queue.Empty:
            pass

        mode = self.config.mode.lower()
        if mode == "auto":
            # Hold off fallback briefly (~0.2s) right after IPC activity to avoid flicker
            if time.monotonic() - self._last_ipc_rx < 0.2:
                try:
                    frame, tag, ts = self.ipc_queue.get(timeout=timeout)
                    self._last_ipc_rx = ts
                    return frame, tag
                except queue.Empty:
                    return None
            try:
                frame, tag, _ = self.sim_queue.get(timeout=timeout)
                return frame, tag
            except queue.Empty:
                return None

        target_q = self.sim_queue if mode == "sim" else self.ipc_queue
        try:
            frame, tag, _ = target_q.get(timeout=timeout)
            return frame, tag
        except queue.Empty:
            return None

    def collect_frames(
        self,
        num_frames: int,
        stop_event: Optional[threading.Event] = None,
        progress_cb=None,
    ) -> np.ndarray:
        """Collect `num_frames` spectrogram vectors into a (num_frames, N_FFT) float32 array."""
        collected = np.empty((num_frames, self.config.n_fft), dtype=np.float32)
        idx = 0
        while idx < num_frames:
            if stop_event is not None and stop_event.is_set():
                return collected[:idx]
            item = self.read_frame(timeout=0.05)
            if item is None:
                continue
            frame, _ = item
            collected[idx] = frame
            idx += 1
            if progress_cb is not None and (idx % 50 == 0 or idx == num_frames):
                progress_cb(idx, num_frames)
        return collected


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="AetherHound GNU Radio DSP Pipeline")
    parser.add_argument(
        "--source",
        choices=("auto", "ipc", "sim", "rtlsdr", "pyaudio"),
        default="auto",
        help="Signal input mode (default: auto).",
    )
    parser.add_argument("--endpoint", default=DEFAULT_IPC_ENDPOINT)
    parser.add_argument("--pub-frames", default=None, help="Optional ZMQ PUB endpoint for FFT frames.")
    parser.add_argument("--duration", type=float, default=5.0, help="Run duration in seconds (0=forever).")
    args = parser.parse_args(argv)

    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    cfg = DSPConfig(
        mode=args.source,
        ipc_endpoint=args.endpoint,
        output_zmq_endpoint=args.pub_frames,
    )
    print(f"[dsp] Starting GNU Radio flowgraph (mode={cfg.mode}, N_FFT={cfg.n_fft})...")
    start_t = time.monotonic()
    count = 0
    with SignalPipeline(cfg) as pipe:
        while not stop_event.is_set():
            if args.duration > 0 and (time.monotonic() - start_t) >= args.duration:
                break
            res = pipe.read_frame(timeout=0.05)
            if res is not None:
                frame, tag = res
                count += 1
                if count % 50 == 0:
                    peak_idx = int(np.argmax(frame))
                    peak_freq = float(pipe.freq_axis_hz[peak_idx])
                    print(
                        f"[dsp] frames={count:5d} src={tag:<12s} "
                        f"peak={peak_freq/1000.0:+6.2f} kHz val={frame[peak_idx]:.3f}"
                    )
    print(f"[dsp] Stopped cleanly after {count} frames.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

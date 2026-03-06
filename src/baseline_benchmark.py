#!/usr/bin/env python3
"""
Baseline Benchmark Pipeline — OpenCV + PyTorch YOLO (CPU decode, GPU infer)

Captures 1 RTSP H264 stream via OpenCV/FFmpeg (software decode).
Runs YOLO inference on GPU via PyTorch.
Writes per-second metrics to results/baseline_metrics.csv.

Usage:
    python baseline_benchmark.py --url rtsp://... --duration 120 --warmup 30
    python baseline_benchmark.py --url rtsp://... --no-display
"""

import argparse
import csv
import os
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import numpy as np
import psutil
import torch

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["QT_LOGGING_RULES"] = "*.debug=false"

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_RTSP = "rtsp://root:Mkvc%402025@192.168.40.40:554/media/stream.sdp?profile=Profile101"
# best.pt and yolo11n.pt are both PCB detectors (nc=1) in this repo.
# For benchmark purposes the class doesn't matter — we compare pipeline metrics, not model accuracy.
DEFAULT_MODEL = str(PROJECT_ROOT / "models" / "best.pt")


# ============================================================================
# GPU MONITOR
# ============================================================================
class GPUMonitor:
    def __init__(self, device_index: int = 0):
        self.handle = None
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception:
                pass

    def read(self):
        """Returns (util_pct, mem_used_mb)"""
        if self.handle is None:
            return 0.0, 0.0
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return float(util.gpu), mem.used / (1024 * 1024)
        except Exception:
            return 0.0, 0.0

    def shutdown(self):
        if PYNVML_AVAILABLE and self.handle:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


# ============================================================================
# RTSP CAPTURE THREAD
# ============================================================================
class RTSPCapture(threading.Thread):
    """
    Continuously reads frames from RTSP stream.
    Timestamps each frame at capture time (cap.read() returns).
    Drops frames if queue is full (latest-frame semantics).
    """

    def __init__(self, url: str, out_queue: Queue, reconnect_delay: float = 2.0):
        super().__init__(daemon=True)
        self.url = url
        self.out_queue = out_queue
        self.reconnect_delay = reconnect_delay
        self.running = False
        self.frames_captured = 0
        self.frames_dropped = 0
        self.cap = None

    def _open(self) -> bool:
        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap = cap
                return True
            cap.release()
            return False
        except Exception:
            return False

    def run(self):
        self.running = True
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                if not self._open():
                    time.sleep(self.reconnect_delay)
                    continue

            ret, frame = self.cap.read()

            if not ret:
                if self.cap:
                    self.cap.release()
                    self.cap = None
                time.sleep(self.reconnect_delay)
                continue

            ts = time.perf_counter()  # high-res capture timestamp
            self.frames_captured += 1

            item = {"frame": frame, "capture_ts": ts}
            try:
                self.out_queue.put_nowait(item)
            except Full:
                # Evict oldest, insert newest
                try:
                    self.out_queue.get_nowait()
                except Empty:
                    pass
                try:
                    self.out_queue.put_nowait(item)
                except Full:
                    pass
                self.frames_dropped += 1

    def stop(self):
        self.running = False
        time.sleep(0.1)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    @property
    def drop_rate(self) -> float:
        if self.frames_captured == 0:
            return 0.0
        return self.frames_dropped / self.frames_captured * 100.0


# ============================================================================
# PREPROCESSOR
# ============================================================================
def letterbox(frame: np.ndarray, size: int = 640) -> np.ndarray:
    """Resize with padding to (size x size), BGR→RGB, contiguous."""
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))


# ============================================================================
# INFERENCE ENGINE
# ============================================================================
class InferenceEngine:
    def __init__(self, model_path: str, device: str = "cuda:0", fp16: bool = True,
                 input_size: int = 640, warmup_iters: int = 20):
        self.device = torch.device(device)
        self.fp16 = fp16
        self.input_size = input_size
        dtype = torch.float16 if fp16 else torch.float32

        print(f"  Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        if fp16:
            self.model.model.half()

        print(f"  Warming up ({warmup_iters} iters)...")
        dummy = torch.zeros(
            (1, 3, input_size, input_size), device=self.device, dtype=dtype
        )
        for _ in range(warmup_iters):
            with torch.amp.autocast("cuda", enabled=fp16):
                with torch.no_grad():
                    _ = self.model.model(dummy)
        torch.cuda.synchronize()

        self.dtype = dtype
        self.cuda_stream = torch.cuda.Stream()
        print("  Inference engine ready.")

    def infer(self, frame_rgb: np.ndarray):
        """
        Single-frame inference.
        Returns (num_detections, inference_time_seconds).
        Includes CPU→GPU transfer in timing (fair: baseline pays this cost).
        """
        dtype = self.dtype

        t0 = time.perf_counter()

        with torch.cuda.stream(self.cuda_stream):
            tensor = (
                torch.from_numpy(frame_rgb)
                .pin_memory()
                .to(self.device, non_blocking=True)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(dtype)
                / 255.0
            )
            with torch.amp.autocast("cuda", enabled=self.fp16):
                with torch.no_grad():
                    results = self.model.model(tensor)
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        # Count detections: use ultralytics postprocessing
        det_count = 0
        try:
            preds = self.model.predictor.postprocess(results, tensor, [])
            if preds and preds[0] is not None:
                det_count = len(preds[0].boxes) if hasattr(preds[0], "boxes") else 0
        except Exception:
            pass

        return det_count, infer_ms


# ============================================================================
# METRICS WINDOW (rolling per second)
# ============================================================================
class MetricsWindow:
    """Thread-safe rolling metrics for 1-second reporting."""

    def __init__(self, window: int = 60):
        self._lock = threading.Lock()
        self._fps_buf: deque = deque(maxlen=window)
        self._lat_buf: deque = deque(maxlen=window)
        self._infer_buf: deque = deque(maxlen=window)
        # frame counter within current second
        self._second_frames: int = 0
        self._second_start: float = time.time()

    def record_frame(self, latency_ms: float, infer_ms: float):
        with self._lock:
            self._second_frames += 1
            self._lat_buf.append(latency_ms)
            self._infer_buf.append(infer_ms)

            now = time.time()
            if now - self._second_start >= 1.0:
                fps = self._second_frames / (now - self._second_start)
                self._fps_buf.append(fps)
                self._second_frames = 0
                self._second_start = now

    def snapshot(self):
        with self._lock:
            fps = float(np.mean(self._fps_buf)) if self._fps_buf else 0.0
            lat = float(np.mean(self._lat_buf)) if self._lat_buf else 0.0
            infer = float(np.mean(self._infer_buf)) if self._infer_buf else 0.0
        return fps, lat, infer


# ============================================================================
# MAIN BENCHMARK
# ============================================================================
class BaselineBenchmark:
    def __init__(self, args):
        self.args = args
        self.running = False
        self.warmup_done = False
        self.gpu = GPUMonitor()

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.csv_path = RESULTS_DIR / "baseline_metrics.csv"

    def run(self):
        self.running = True
        args = self.args

        print("\n" + "=" * 60)
        print("  BASELINE BENCHMARK  (OpenCV + PyTorch YOLO)")
        print("=" * 60)
        print(f"  RTSP URL  : {args.url}")
        print(f"  Model     : {args.model}")
        print(f"  Warmup    : {args.warmup}s")
        print(f"  Duration  : {args.duration}s")
        print(f"  FP16      : {not args.fp32}")
        print(f"  Output CSV: {self.csv_path}")
        print("=" * 60 + "\n")

        # Start capture
        frame_queue: Queue = Queue(maxsize=4)
        capture = RTSPCapture(args.url, frame_queue)
        capture.start()
        time.sleep(1.0)  # let connection settle

        # Load inference engine
        engine = InferenceEngine(
            model_path=args.model,
            fp16=not args.fp32,
            warmup_iters=20,
        )

        metrics = MetricsWindow()
        csv_rows = []

        # ---- WARMUP PHASE ----
        print(f"🔥 Warmup {args.warmup}s …")
        warmup_end = time.time() + args.warmup
        while self.running and time.time() < warmup_end:
            try:
                item = frame_queue.get(timeout=0.5)
            except Empty:
                continue
            frame = item["frame"]
            processed = letterbox(frame, engine.input_size)
            engine.infer(processed)  # discard results
        self.warmup_done = True
        # Reset drop counters after warmup
        capture.frames_captured = 0
        capture.frames_dropped = 0

        print(f"✅ Warmup done. Starting measurement ({args.duration}s)…\n")

        # ---- MEASUREMENT PHASE ----
        measure_start = time.time()
        measure_end = measure_start + args.duration
        total_frames_processed = 0
        interval_start = measure_start
        interval_frames = 0

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp_s",
                    "fps",
                    "latency_ms",
                    "cpu_pct",
                    "gpu_util_pct",
                    "gpu_mem_mb",
                    "infer_ms",
                    "drop_rate_pct",
                ]
            )

            next_csv_write = time.time() + 1.0  # write a row every second

            while self.running and time.time() < measure_end:
                try:
                    item = frame_queue.get(timeout=0.2)
                except Empty:
                    continue

                capture_ts = item["capture_ts"]
                frame = item["frame"]

                # Preprocess (CPU)
                processed = letterbox(frame, engine.input_size)

                # Infer (GPU)
                det_count, infer_ms = engine.infer(processed)

                infer_done_ts = time.perf_counter()
                latency_ms = (infer_done_ts - capture_ts) * 1000.0

                metrics.record_frame(latency_ms, infer_ms)
                total_frames_processed += 1
                interval_frames += 1

                # Write one CSV row per second
                now = time.time()
                if now >= next_csv_write:
                    elapsed = now - measure_start
                    fps_interval = interval_frames / (now - interval_start)
                    avg_fps, avg_lat, avg_infer = metrics.snapshot()
                    gpu_util, gpu_mem = self.gpu.read()
                    cpu_pct = psutil.cpu_percent(interval=None)
                    drop = capture.drop_rate

                    row = [
                        f"{elapsed:.1f}",
                        f"{fps_interval:.2f}",
                        f"{avg_lat:.2f}",
                        f"{cpu_pct:.1f}",
                        f"{gpu_util:.1f}",
                        f"{gpu_mem:.1f}",
                        f"{avg_infer:.2f}",
                        f"{drop:.2f}",
                    ]
                    writer.writerow(row)
                    f.flush()
                    csv_rows.append(row)

                    if not args.no_display:
                        print(
                            f"\r[{elapsed:6.1f}s] "
                            f"FPS:{fps_interval:5.1f} | "
                            f"Lat:{avg_lat:6.1f}ms | "
                            f"Infer:{avg_infer:6.1f}ms | "
                            f"CPU:{cpu_pct:4.0f}% | "
                            f"GPU:{gpu_util:3.0f}% {gpu_mem:6.0f}MB | "
                            f"Drop:{drop:4.1f}%",
                            end="",
                            flush=True,
                        )

                    interval_frames = 0
                    interval_start = now
                    next_csv_write = now + 1.0

        # ---- SUMMARY ----
        total_elapsed = time.time() - measure_start
        overall_fps = total_frames_processed / total_elapsed if total_elapsed > 0 else 0

        if csv_rows:
            fps_vals = [float(r[1]) for r in csv_rows]
            lat_vals = [float(r[2]) for r in csv_rows]
            cpu_vals = [float(r[3]) for r in csv_rows]
            gpu_vals = [float(r[4]) for r in csv_rows]
            infer_vals = [float(r[6]) for r in csv_rows]
            drop_vals = [float(r[7]) for r in csv_rows]

            print(f"\n\n{'='*60}")
            print("  BASELINE RESULTS SUMMARY")
            print(f"{'='*60}")
            print(f"  Pipeline      : OpenCV FFmpeg decode + PyTorch YOLO FP16")
            print(f"  Duration      : {total_elapsed:.1f}s")
            print(f"  Total frames  : {total_frames_processed}")
            print(f"  Avg FPS       : {np.mean(fps_vals):.1f}  (min {np.min(fps_vals):.1f} / max {np.max(fps_vals):.1f})")
            print(f"  Avg Latency   : {np.mean(lat_vals):.1f} ms")
            print(f"  Avg Infer     : {np.mean(infer_vals):.1f} ms")
            print(f"  Avg CPU       : {np.mean(cpu_vals):.1f}%")
            print(f"  Avg GPU util  : {np.mean(gpu_vals):.1f}%")
            print(f"  Avg Drop rate : {np.mean(drop_vals):.2f}%")
            print(f"{'='*60}")
            print(f"  CSV saved to  : {self.csv_path}")
            print(f"{'='*60}\n")

        # Cleanup
        capture.stop()
        capture.join(timeout=5)
        self.gpu.shutdown()
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Baseline OpenCV+PyTorch benchmark")
    p.add_argument(
        "--url",
        default=DEFAULT_RTSP,
        help="RTSP stream URL",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="YOLO model path (.pt)",
    )
    p.add_argument("--warmup", type=int, default=30, help="Warmup seconds (default: 30)")
    p.add_argument("--duration", type=int, default=120, help="Measure seconds (default: 120)")
    p.add_argument("--fp32", action="store_true", help="Use FP32 (default: FP16)")
    p.add_argument("--no-display", action="store_true", help="Suppress live stats output")
    return p.parse_args()


def main():
    args = parse_args()
    bench = BaselineBenchmark(args)

    def _handle_signal(sig, frame):
        print("\n⚠️  Interrupted — stopping measurement.")
        bench.running = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    bench.run()


if __name__ == "__main__":
    main()

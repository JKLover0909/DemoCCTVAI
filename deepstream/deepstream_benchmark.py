#!/usr/bin/env python3
"""
DeepStream 8.0 Benchmark Pipeline — NVDEC + TensorRT YOLO (GPU decode, GPU infer)
NO pyds required — uses pure GStreamer Python (gi.repository.Gst) + appsink.

Pipeline:
  rtspsrc → rtph264depay → h264parse →
  nvv4l2decoder (NVDEC) →
  nvstreammux →
  nvinfer (TensorRT) →
  nvtracker (NvDCF) →
  appsink  ← timestamps every frame for latency + FPS measurement

Usage (inside container, via run_benchmark.sh):
    python3 deepstream_benchmark.py --url rtsp://... --duration 120 --warmup 30
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

import psutil

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GLib", "2.0")
from gi.repository import GLib, Gst

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# ============================================================================
# PATHS (inside container)
# ============================================================================
APP_DIR    = Path("/app")
MODELS_DIR = Path("/models")
RESULTS_DIR = Path("/results")
CONFIG_DIR  = APP_DIR / "config"
LABELS_FILE = APP_DIR / "labels.txt"

INFER_CONFIG = str(CONFIG_DIR / "config_infer_primary.txt")

TRACKER_LIB = (
    "/opt/nvidia/deepstream/deepstream/lib/"
    "libnvds_nvmultiobjecttracker.so"
)
TRACKER_YML = (
    "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/"
    "config_tracker_NvDCF_perf.yml"
)


# ============================================================================
# GPU MONITOR
# ============================================================================
class GPUMonitor:
    def __init__(self):
        self.handle = None
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                pass

    def read(self):
        if self.handle is None:
            return 0.0, 0.0
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
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
# SHARED STATE
# ============================================================================
class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self.frame_count_interval: int = 0
        self.latency_buf: deque = deque(maxlen=300)
        self.warmup_done: bool = False
        self.measure_start: float = 0.0
        self.stop_event = threading.Event()

    def record_frame(self, latency_ms: float):
        with self._lock:
            if not self.warmup_done:
                return
            self.frame_count_interval += 1
            if latency_ms > 0:
                self.latency_buf.append(latency_ms)

    def consume_frames(self):
        with self._lock:
            n = self.frame_count_interval
            self.frame_count_interval = 0
        return n

    def avg_latency(self):
        with self._lock:
            if not self.latency_buf:
                return 0.0
            return sum(self.latency_buf) / len(self.latency_buf)

    def mark_warmup_done(self):
        with self._lock:
            self.warmup_done = True
            self.measure_start = time.time()


# ============================================================================
# BUILD PIPELINE
# ============================================================================
def build_pipeline(rtsp_url: str):
    pipeline = Gst.Pipeline.new("ds-benchmark")

    def make(factory, name):
        el = Gst.ElementFactory.make(factory, name)
        if not el:
            raise RuntimeError(f"Could not create element: {factory} ({name})")
        pipeline.add(el)
        return el

    src       = make("rtspsrc",       "src")
    depay     = make("rtph264depay",  "depay")
    parser    = make("h264parse",     "parser")
    decoder   = make("nvv4l2decoder", "decoder")
    streammux = make("nvstreammux",   "mux")
    nvinfer   = make("nvinfer",       "infer")
    tracker   = make("nvtracker",     "tracker")
    sink      = make("appsink",       "sink")

    src.set_property("location",        rtsp_url)
    src.set_property("protocols",       4)
    src.set_property("latency",         100)
    src.set_property("drop-on-latency", True)

    streammux.set_property("batch-size",           1)
    streammux.set_property("width",                1280)
    streammux.set_property("height",               960)
    streammux.set_property("batched-push-timeout", 4_000_000)
    streammux.set_property("live-source",          True)
    streammux.set_property("enable-padding",       False)

    nvinfer.set_property("config-file-path", INFER_CONFIG)

    tracker.set_property("ll-lib-file",        TRACKER_LIB)
    tracker.set_property("ll-config-file",     TRACKER_YML)
    tracker.set_property("tracker-width",  640)
    tracker.set_property("tracker-height", 384)
    tracker.set_property("gpu-id",          0)

    sink.set_property("emit-signals", True)
    sink.set_property("sync",         False)
    sink.set_property("async",        False)
    sink.set_property("max-buffers",  5)
    sink.set_property("drop",         True)

    nvinfer.link(tracker)
    tracker.link(sink)
    depay.link(parser)
    parser.link(decoder)

    def on_src_pad(element, pad):
        caps = pad.get_current_caps()
        if caps and "application/x-rtp" in caps.to_string():
            sp = depay.get_static_pad("sink")
            if not sp.is_linked():
                pad.link(sp)

    src.connect("pad-added", on_src_pad)

    def on_dec_pad(element, pad):
        mux_sink = streammux.request_pad_simple("sink_0")
        if mux_sink and not pad.is_linked():
            pad.link(mux_sink)

    decoder.connect("pad-added", on_dec_pad)

    return pipeline, sink


# ============================================================================
# CSV WRITER THREAD
# ============================================================================
def csv_writer_thread(state: SharedState, gpu: GPUMonitor,
                      csv_path: Path, display: bool, duration_s: int):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_s", "fps", "latency_ms", "cpu_pct",
            "gpu_util_pct", "gpu_mem_mb", "infer_ms", "drop_rate_pct",
        ])
        t_interval   = time.time()
        measure_start = None

        while not state.stop_event.is_set():
            time.sleep(1.0)
            if not state.warmup_done:
                continue
            if measure_start is None:
                measure_start = time.time()

            now     = time.time()
            elapsed = now - measure_start
            dt      = now - t_interval
            n       = state.consume_frames()
            fps     = n / dt if dt > 0 else 0.0
            lat     = state.avg_latency()
            gu, gm  = gpu.read()
            cpu     = psutil.cpu_percent(interval=None)

            row = [
                f"{elapsed:.1f}", f"{fps:.2f}", f"{lat:.2f}",
                f"{cpu:.1f}", f"{gu:.1f}", f"{gm:.1f}",
                "0.00", "0.00",
            ]
            writer.writerow(row)
            f.flush()

            if display:
                print(
                    f"\r[{elapsed:6.1f}s] "
                    f"FPS:{fps:5.1f} | "
                    f"Lat:{lat:6.1f}ms | "
                    f"CPU:{cpu:4.0f}% | "
                    f"GPU:{gu:3.0f}% {gm:6.0f}MB",
                    end="", flush=True,
                )

            t_interval = now
            if elapsed >= duration_s:
                state.stop_event.set()


# ============================================================================
# MAIN BENCHMARK
# ============================================================================
class DeepStreamBenchmark:
    def __init__(self, args):
        self.args  = args
        self.gpu   = GPUMonitor()
        self.state = SharedState()
        self.loop  = None
        self.pipe  = None

    def run(self):
        args     = self.args
        csv_path = RESULTS_DIR / "deepstream_metrics.csv"

        print("\n" + "=" * 60)
        print("  DEEPSTREAM 8.0 BENCHMARK  (NVDEC + TensorRT YOLO)")
        print("=" * 60)
        print(f"  RTSP URL  : {args.url}")
        print(f"  Config    : {INFER_CONFIG}")
        print(f"  Warmup    : {args.warmup}s")
        print(f"  Duration  : {args.duration}s")
        print(f"  Output CSV: {csv_path}")
        print("=" * 60)

        num_classes = 1
        if LABELS_FILE.exists():
            with open(LABELS_FILE) as lf:
                lines = [l.strip() for l in lf if l.strip()]
                num_classes = len(lines)
            print(f"  Classes ({num_classes}): {', '.join(lines)}")
        print()

        Gst.init(None)

        try:
            self.pipe, appsink = build_pipeline(args.url)
        except RuntimeError as e:
            print(f"ERROR building pipeline: {e}")
            sys.exit(1)

        wall_start_ns = time.perf_counter_ns()

        def on_new_sample(sink):
            sample = sink.emit("pull-sample")
            if sample is None:
                return Gst.FlowReturn.ERROR
            buf   = sample.get_buffer()
            now_ns = time.perf_counter_ns()
            pts   = buf.pts
            if pts != Gst.CLOCK_TIME_NONE and pts > 0:
                wall_ms = (now_ns - wall_start_ns) / 1_000_000.0
                pts_ms  = pts / 1_000_000.0
                lat     = max(0.0, min(wall_ms - pts_ms, 2000.0))
            else:
                lat = 0.0
            self.state.record_frame(lat)
            return Gst.FlowReturn.OK

        appsink.connect("new-sample", on_new_sample)

        writer_t = threading.Thread(
            target=csv_writer_thread,
            args=(self.state, self.gpu, csv_path, not args.no_display, args.duration),
            daemon=True,
        )
        writer_t.start()

        self.loop = GLib.MainLoop()
        bus = self.pipe.get_bus()
        bus.add_signal_watch()

        def on_message(bus, msg):
            t = msg.type
            if t == Gst.MessageType.EOS:
                print("\n  EOS.")
                self.loop.quit()
            elif t == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                print(f"\n  GStreamer ERROR: {err.message}")
                if dbg:
                    print(f"  Debug: {dbg}")
                self.state.stop_event.set()
                self.loop.quit()
            elif t == Gst.MessageType.WARNING:
                w, _ = msg.parse_warning()
                print(f"\n  GStreamer WARNING: {w.message}")
            return True

        bus.connect("message", on_message)

        def on_warmup_done():
            print(f"\n✅ Warmup done. Measuring {args.duration}s …\n")
            self.state.mark_warmup_done()
            return False

        def on_measure_done():
            print(f"\n⏹  Measurement complete.")
            self.state.stop_event.set()
            self.loop.quit()
            return False

        GLib.timeout_add_seconds(args.warmup, on_warmup_done)
        GLib.timeout_add_seconds(args.warmup + args.duration, on_measure_done)

        print(f"🔥 Warmup {args.warmup}s …")
        self.pipe.set_state(Gst.State.PLAYING)

        try:
            self.loop.run()
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted.")
            self.state.stop_event.set()

        self.pipe.set_state(Gst.State.NULL)
        writer_t.join(timeout=5)
        self.gpu.shutdown()
        self._print_summary(csv_path)

    def _print_summary(self, csv_path: Path):
        if not csv_path.exists():
            return
        import numpy as np
        rows = []
        with open(csv_path) as f:
            import csv as _csv
            for row in _csv.DictReader(f):
                rows.append(row)
        if not rows:
            print("CSV empty.")
            return
        fps = [float(r["fps"])          for r in rows]
        lat = [float(r["latency_ms"])   for r in rows]
        cpu = [float(r["cpu_pct"])      for r in rows]
        gu  = [float(r["gpu_util_pct"]) for r in rows]
        gm  = [float(r["gpu_mem_mb"])   for r in rows]
        print(f"\n{'='*60}")
        print("  DEEPSTREAM RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  Pipeline   : NVDEC (GPU decode) + TensorRT YOLO FP16")
        print(f"  Duration   : {float(rows[-1]['timestamp_s']):.1f}s")
        print(f"  Avg FPS    : {np.mean(fps):.1f}  (min {np.min(fps):.1f} / max {np.max(fps):.1f})")
        print(f"  Avg Latency: {np.mean(lat):.1f} ms")
        print(f"  Avg CPU    : {np.mean(cpu):.1f}%")
        print(f"  Avg GPU    : {np.mean(gu):.1f}%  Mem: {np.mean(gm):.0f} MB")
        print(f"{'='*60}")
        print(f"  CSV: {csv_path}")
        print(f"{'='*60}\n")


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url",
        default="rtsp://root:Mkvc%402025@192.168.40.40:554/media/stream.sdp?profile=Profile101")
    p.add_argument("--warmup",     type=int, default=30)
    p.add_argument("--duration",   type=int, default=120)
    p.add_argument("--no-display", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    bench = DeepStreamBenchmark(args)

    def _sig(sig, frame):
        bench.state.stop_event.set()
        if bench.loop:
            bench.loop.quit()

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)
    bench.run()


if __name__ == "__main__":
    main()

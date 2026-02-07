#!/usr/bin/env python3
"""
Multi-Camera YOLO Inference Benchmark Pipeline

Simulates 10-30 MJPEG camera streams and measures GPU inference capacity using YOLOv11n.
Optimized for maximum GPU throughput with AMP, pinned memory, and async CUDA streams.

Usage:
    python benchmark_multicam.py --streams 16 --duration 60
"""

import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false"

import argparse
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import requests
import torch
from requests.auth import HTTPDigestAuth

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

from ultralytics import YOLO


# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class Config:
    """Pipeline configuration"""
    # Stream settings
    num_streams: int = 16
    stream_type: str = "video"  # "mjpeg" or "video"
    stream_url: str = "/home/jkl/Code/DemoCCTVAI/assets/videos/data1.mp4"
    stream_username: str = "root"
    stream_password: str = "Mkvc@2025"
    
    # Queue settings
    queue_size: int = 2  # Small queue = drop old frames
    
    # Batch settings
    batch_size_min: int = 4
    batch_size_max: int = 16
    batch_timeout_ms: float = 50.0
    
    # Inference settings
    model_path: str = "/home/jkl/Code/DemoCCTVAI/models/best.pt"
    input_size: int = 640
    use_fp16: bool = True
    device: str = "cuda:0"
    
    # Benchmark settings
    duration_seconds: int = 60
    stats_interval: float = 1.0


# ============================================================================
# METRICS TRACKER
# ============================================================================
@dataclass
class StreamMetrics:
    """Per-stream metrics"""
    frames_captured: int = 0
    frames_dropped: int = 0
    frames_processed: int = 0
    last_capture_time: float = 0.0
    latencies: deque = field(default_factory=lambda: deque(maxlen=100))


class MetricsTracker:
    """Collects and reports pipeline metrics"""
    
    def __init__(self, num_streams: int):
        self.num_streams = num_streams
        self.stream_metrics: Dict[int, StreamMetrics] = {
            i: StreamMetrics() for i in range(num_streams)
        }
        self.total_inferences = 0
        self.total_detections = 0
        self.inference_times: deque = deque(maxlen=100)
        self.batch_sizes: deque = deque(maxlen=100)
        self.start_time = time.time()
        self.lock = threading.Lock()
        
        # Initialize pynvml for GPU metrics
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.gpu_handle = None
        else:
            self.gpu_handle = None
    
    def record_capture(self, stream_id: int, dropped: bool = False):
        with self.lock:
            m = self.stream_metrics[stream_id]
            m.frames_captured += 1
            m.last_capture_time = time.time()
            if dropped:
                m.frames_dropped += 1
    
    def record_inference(self, batch_size: int, inference_time: float, 
                         detections: int, stream_ids: List[int], latencies: List[float]):
        with self.lock:
            self.total_inferences += 1
            self.total_detections += detections
            self.inference_times.append(inference_time)
            self.batch_sizes.append(batch_size)
            
            for sid, lat in zip(stream_ids, latencies):
                self.stream_metrics[sid].frames_processed += 1
                self.stream_metrics[sid].latencies.append(lat)
    
    def get_gpu_utilization(self) -> Tuple[float, float]:
        """Returns (GPU util %, memory util %)"""
        if self.gpu_handle is None:
            return 0.0, 0.0
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return util.gpu, (mem.used / mem.total) * 100
        except Exception:
            return 0.0, 0.0
    
    def get_stats(self) -> dict:
        with self.lock:
            elapsed = time.time() - self.start_time
            
            # Per-stream FPS
            stream_fps = {}
            for sid, m in self.stream_metrics.items():
                stream_fps[sid] = m.frames_processed / elapsed if elapsed > 0 else 0
            
            # Total captures and drops
            total_captured = sum(m.frames_captured for m in self.stream_metrics.values())
            total_dropped = sum(m.frames_dropped for m in self.stream_metrics.values())
            total_processed = sum(m.frames_processed for m in self.stream_metrics.values())
            
            # Latency stats
            all_latencies = []
            for m in self.stream_metrics.values():
                all_latencies.extend(m.latencies)
            avg_latency = np.mean(all_latencies) * 1000 if all_latencies else 0
            
            # Inference stats
            avg_inf_time = np.mean(self.inference_times) * 1000 if self.inference_times else 0
            avg_batch = np.mean(self.batch_sizes) if self.batch_sizes else 0
            
            gpu_util, mem_util = self.get_gpu_utilization()
            
            return {
                "elapsed": elapsed,
                "total_captured": total_captured,
                "total_dropped": total_dropped,
                "total_processed": total_processed,
                "drop_rate": total_dropped / total_captured * 100 if total_captured > 0 else 0,
                "overall_fps": total_processed / elapsed if elapsed > 0 else 0,
                "avg_stream_fps": np.mean(list(stream_fps.values())),
                "avg_latency_ms": avg_latency,
                "avg_inference_ms": avg_inf_time,
                "avg_batch_size": avg_batch,
                "total_inferences": self.total_inferences,
                "total_detections": self.total_detections,
                "gpu_util": gpu_util,
                "mem_util": mem_util,
                "cpu_util": psutil.cpu_percent(),
            }
    
    def shutdown(self):
        if PYNVML_AVAILABLE and self.gpu_handle:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


# ============================================================================
# FRAME PREPROCESSOR
# ============================================================================
class FramePreprocessor:
    """Preprocesses frames for inference"""
    
    def __init__(self, target_size: int = 640):
        self.target_size = target_size
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Resize and convert BGR to RGB, return contiguous array"""
        # Resize maintaining aspect ratio with padding
        h, w = frame.shape[:2]
        scale = self.target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        pad_x = (self.target_size - new_w) // 2
        pad_y = (self.target_size - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        # BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        return np.ascontiguousarray(rgb)


# ============================================================================
# MJPEG STREAM WORKER
# ============================================================================
class MJPEGStreamWorker(threading.Thread):
    """Thread-based MJPEG stream capture with non-blocking queue push"""
    
    def __init__(self, stream_id: int, url: str, username: str, password: str,
                 output_queue: Queue, preprocessor: FramePreprocessor,
                 metrics: MetricsTracker):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.url = url
        self.auth = HTTPDigestAuth(username, password)
        self.output_queue = output_queue
        self.preprocessor = preprocessor
        self.metrics = metrics
        self.running = False
        self.headers = {"User-Agent": "Mozilla/5.0 BenchmarkBot"}
    
    def run(self):
        self.running = True
        bytes_buffer = b''
        chunk_size = 4096
        max_buffer = 2 * 1024 * 1024  # 2MB
        
        while self.running:
            try:
                response = requests.get(
                    self.url, stream=True, auth=self.auth,
                    headers=self.headers, timeout=10
                )
                
                if response.status_code != 200:
                    time.sleep(1)
                    continue
                
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not self.running:
                        break
                    
                    bytes_buffer += chunk
                    
                    # Limit buffer size
                    if len(bytes_buffer) > max_buffer:
                        bytes_buffer = bytes_buffer[-max_buffer:]
                    
                    # Find JPEG markers
                    start = bytes_buffer.find(b'\xff\xd8')
                    end = bytes_buffer.find(b'\xff\xd9', start)
                    
                    if start != -1 and end != -1:
                        jpg_data = bytes_buffer[start:end+2]
                        bytes_buffer = bytes_buffer[end+2:]
                        
                        # Decode frame
                        frame = cv2.imdecode(
                            np.frombuffer(jpg_data, dtype=np.uint8),
                            cv2.IMREAD_COLOR
                        )
                        
                        if frame is not None:
                            # Preprocess in capture thread
                            processed = self.preprocessor.process(frame)
                            capture_time = time.time()
                            
                            # Non-blocking push - drop if full
                            try:
                                self.output_queue.put_nowait({
                                    "stream_id": self.stream_id,
                                    "frame": processed,
                                    "timestamp": capture_time
                                })
                                self.metrics.record_capture(self.stream_id, dropped=False)
                            except Full:
                                self.metrics.record_capture(self.stream_id, dropped=True)
                
                response.close()
                
            except Exception as e:
                if self.running:
                    time.sleep(1)
        
    
    def stop(self):
        self.running = False


# ============================================================================
# VIDEO FILE WORKER
# ============================================================================
class VideoFileWorker(threading.Thread):
    """Thread-based video file capture with non-blocking queue push"""
    
    def __init__(self, stream_id: int, video_path: str,
                 output_queue: Queue, preprocessor: FramePreprocessor,
                 metrics: MetricsTracker, loop: bool = True):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.video_path = video_path
        self.output_queue = output_queue
        self.preprocessor = preprocessor
        self.metrics = metrics
        self.loop = loop
        self.running = False
    
    def run(self):
        self.running = True
        
        while self.running:
            cap = cv2.VideoCapture(self.video_path)
            
            if not cap.isOpened():
                print(f"⚠️  Stream {self.stream_id}: Cannot open video {self.video_path}")
                time.sleep(1)
                continue
            
            while self.running:
                ret, frame = cap.read()
                
                if not ret:
                    # End of video
                    if self.loop:
                        break  # Restart from beginning
                    else:
                        self.running = False
                        break
                
                # Preprocess in capture thread
                processed = self.preprocessor.process(frame)
                capture_time = time.time()
                
                # Non-blocking push - drop if full
                try:
                    self.output_queue.put_nowait({
                        "stream_id": self.stream_id,
                        "frame": processed,
                        "timestamp": capture_time
                    })
                    self.metrics.record_capture(self.stream_id, dropped=False)
                except Full:
                    self.metrics.record_capture(self.stream_id, dropped=True)
            
            cap.release()
            
            if not self.loop:
                break
    
    def stop(self):
        self.running = False


# ============================================================================
# BATCH AGGREGATOR
# ============================================================================
class BatchAggregator:
    """Collects frames and forms dynamic batches"""
    
    def __init__(self, queues: List[Queue], batch_size_min: int, 
                 batch_size_max: int, timeout_ms: float):
        self.queues = queues
        self.batch_size_min = batch_size_min
        self.batch_size_max = batch_size_max
        self.timeout = timeout_ms / 1000.0
        self.pending: List[dict] = []
        self.last_flush = time.time()
    
    def get_batch(self) -> Optional[Tuple[np.ndarray, List[int], List[float]]]:
        """
        Returns (batch_array, stream_ids, timestamps) or None if no frames.
        Batch is shape (N, H, W, 3) as uint8 RGB.
        """
        # Collect from all queues (round-robin)
        for q in self.queues:
            try:
                item = q.get_nowait()
                self.pending.append(item)
            except Empty:
                pass
            
            # Stop if we hit max batch
            if len(self.pending) >= self.batch_size_max:
                break
        
        now = time.time()
        should_flush = (
            len(self.pending) >= self.batch_size_max or
            (len(self.pending) >= self.batch_size_min and 
             now - self.last_flush >= self.timeout) or
            (len(self.pending) > 0 and now - self.last_flush >= self.timeout * 2)
        )
        
        if should_flush and self.pending:
            batch_items = self.pending[:self.batch_size_max]
            self.pending = self.pending[self.batch_size_max:]
            self.last_flush = now
            
            frames = np.stack([item["frame"] for item in batch_items])
            stream_ids = [item["stream_id"] for item in batch_items]
            timestamps = [item["timestamp"] for item in batch_items]
            
            return frames, stream_ids, timestamps
        
        return None


# ============================================================================
# INFERENCE ENGINE
# ============================================================================
class InferenceEngine:
    """GPU-optimized YOLO inference with AMP and async streams"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load model
        print(f"📦 Loading model: {config.model_path}")
        self.model = YOLO(config.model_path)
        self.model.to(self.device)
        
        if config.use_fp16:
            self.model.model.half()
        
        # Warmup
        print("🔥 Warming up model...")
        dummy = torch.zeros((1, 3, config.input_size, config.input_size), 
                            device=self.device, 
                            dtype=torch.float16 if config.use_fp16 else torch.float32)
        for _ in range(10):
            with torch.amp.autocast('cuda', enabled=config.use_fp16):
                _ = self.model.model(dummy)
        torch.cuda.synchronize()
        
        # Create CUDA stream for async
        self.cuda_stream = torch.cuda.Stream()
        
        # Preallocate buffer
        self.batch_buffer = torch.empty(
            (config.batch_size_max, 3, config.input_size, config.input_size),
            device=self.device,
            dtype=torch.float16 if config.use_fp16 else torch.float32
        )
        
        print("✅ Inference engine ready")
    
    def infer(self, batch: np.ndarray) -> Tuple[int, float]:
        """
        Run inference on batch.
        Args:
            batch: numpy array shape (N, H, W, 3) uint8 RGB
        Returns:
            (total_detections, inference_time_seconds)
        """
        batch_size = batch.shape[0]
        
        # Convert to tensor: (N, H, W, 3) -> (N, 3, H, W) normalized
        with torch.cuda.stream(self.cuda_stream):
            # Use pinned memory for faster transfer
            tensor = torch.from_numpy(batch).pin_memory()
            tensor = tensor.to(self.device, non_blocking=True)
            tensor = tensor.permute(0, 3, 1, 2).contiguous()
            tensor = tensor.to(torch.float16 if self.config.use_fp16 else torch.float32)
            tensor = tensor / 255.0
            
            # Copy to preallocated buffer
            self.batch_buffer[:batch_size].copy_(tensor)
            
            start = time.perf_counter()
            
            with torch.amp.autocast('cuda', enabled=self.config.use_fp16):
                with torch.no_grad():
                    results = self.model.model(self.batch_buffer[:batch_size])
            
            torch.cuda.synchronize()
            inference_time = time.perf_counter() - start
        
        # Count detections (simplified - just count boxes)
        total_detections = 0
        if isinstance(results, (list, tuple)):
            for r in results:
                if hasattr(r, 'shape') and len(r.shape) >= 2:
                    # Assuming detection output format
                    total_detections += r.shape[0] if r.shape[0] < 1000 else 0
        
        return total_detections, inference_time


# ============================================================================
# MAIN PIPELINE
# ============================================================================
class BenchmarkPipeline:
    """Main orchestrator for the benchmark"""
    
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        
        # Initialize components
        self.metrics = MetricsTracker(config.num_streams)
        self.preprocessor = FramePreprocessor(config.input_size)
        
        # Create queues and workers
        self.queues: List[Queue] = []
        self.workers: List[MJPEGStreamWorker] = []
        
        for i in range(config.num_streams):
            q = Queue(maxsize=config.queue_size)
            self.queues.append(q)
            
            # Choose worker type based on stream_type
            if config.stream_type == "video":
                worker = VideoFileWorker(
                    stream_id=i,
                    video_path=config.stream_url,
                    output_queue=q,
                    preprocessor=self.preprocessor,
                    metrics=self.metrics,
                    loop=True
                )
            else:  # mjpeg
                worker = MJPEGStreamWorker(
                    stream_id=i,
                    url=config.stream_url,
                    username=config.stream_username,
                    password=config.stream_password,
                    output_queue=q,
                    preprocessor=self.preprocessor,
                    metrics=self.metrics
                )
            self.workers.append(worker)
        
        # Batch aggregator
        self.aggregator = BatchAggregator(
            self.queues,
            config.batch_size_min,
            config.batch_size_max,
            config.batch_timeout_ms
        )
        
        # Inference engine
        self.engine = InferenceEngine(config)
        
        # Stats thread
        self.stats_thread: Optional[threading.Thread] = None
    
    def _print_stats(self):
        """Print live stats every interval"""
        while self.running:
            time.sleep(self.config.stats_interval)
            if not self.running:
                break
            
            stats = self.metrics.get_stats()
            
            print(f"\r[{stats['elapsed']:.1f}s] "
                  f"FPS: {stats['overall_fps']:.1f} | "
                  f"Latency: {stats['avg_latency_ms']:.1f}ms | "
                  f"Infer: {stats['avg_inference_ms']:.1f}ms | "
                  f"Batch: {stats['avg_batch_size']:.1f} | "
                  f"Drop: {stats['drop_rate']:.1f}% | "
                  f"GPU: {stats['gpu_util']:.0f}% | "
                  f"Mem: {stats['mem_util']:.0f}%", 
                  end="", flush=True)
    
    def run(self):
        """Run the benchmark"""
        self.running = True
        
        print(f"\n{'='*60}")
        print(f"🎥 Starting benchmark with {self.config.num_streams} streams")
        print(f"📊 Batch size: {self.config.batch_size_min}-{self.config.batch_size_max}")
        print(f"⏱️  Duration: {self.config.duration_seconds}s")
        print(f"{'='*60}\n")
        
        # Start workers
        for worker in self.workers:
            worker.start()
        
        # Start stats thread
        self.stats_thread = threading.Thread(target=self._print_stats, daemon=True)
        self.stats_thread.start()
        
        # Main inference loop
        end_time = time.time() + self.config.duration_seconds
        
        try:
            while self.running and time.time() < end_time:
                batch_data = self.aggregator.get_batch()
                
                if batch_data is not None:
                    frames, stream_ids, timestamps = batch_data
                    
                    # Run inference
                    detections, inf_time = self.engine.infer(frames)
                    
                    # Calculate latencies
                    now = time.time()
                    latencies = [now - ts for ts in timestamps]
                    
                    # Record metrics
                    self.metrics.record_inference(
                        batch_size=len(stream_ids),
                        inference_time=inf_time,
                        detections=detections,
                        stream_ids=stream_ids,
                        latencies=latencies
                    )
                else:
                    # Small sleep to avoid busy loop
                    time.sleep(0.001)
                    
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        
        self.shutdown()
    
    def shutdown(self):
        """Clean shutdown"""
        print("\n\n🛑 Shutting down...")
        self.running = False
        
        # Stop workers
        for worker in self.workers:
            worker.stop()
        
        # Wait for workers
        for worker in self.workers:
            worker.join(timeout=2)
        
        # Final stats
        stats = self.metrics.get_stats()
        print(f"\n{'='*60}")
        print("📈 FINAL RESULTS")
        print(f"{'='*60}")
        print(f"  Duration:        {stats['elapsed']:.1f}s")
        print(f"  Streams:         {self.config.num_streams}")
        print(f"  Frames captured: {stats['total_captured']}")
        print(f"  Frames dropped:  {stats['total_dropped']} ({stats['drop_rate']:.1f}%)")
        print(f"  Frames processed:{stats['total_processed']}")
        print(f"  Overall FPS:     {stats['overall_fps']:.1f}")
        print(f"  Avg stream FPS:  {stats['avg_stream_fps']:.2f}")
        print(f"  Avg latency:     {stats['avg_latency_ms']:.1f}ms")
        print(f"  Avg inference:   {stats['avg_inference_ms']:.1f}ms")
        print(f"  Avg batch size:  {stats['avg_batch_size']:.1f}")
        print(f"  Total inferences:{stats['total_inferences']}")
        print(f"  GPU utilization: {stats['gpu_util']:.0f}%")
        print(f"  GPU memory:      {stats['mem_util']:.0f}%")
        print(f"{'='*60}\n")
        
        self.metrics.shutdown()


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Camera YOLO Inference Benchmark"
    )
    parser.add_argument("--streams", type=int, default=16,
                        help="Number of camera streams (default: 16)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Benchmark duration in seconds (default: 60)")
    parser.add_argument("--type", type=str, choices=["video", "mjpeg"], default="video",
                        help="Stream type: video file or mjpeg (default: video)")
    parser.add_argument("--url", type=str, 
                        default="/home/jkl/Code/DemoCCTVAI/assets/videos/data1.mp4",
                        help="Video file path or MJPEG stream URL")
    parser.add_argument("--model", type=str, default="/home/jkl/Code/DemoCCTVAI/models/best.pt",
                        help="YOLO model path")
    parser.add_argument("--batch-min", type=int, default=4,
                        help="Minimum batch size (default: 4)")
    parser.add_argument("--batch-max", type=int, default=16,
                        help="Maximum batch size (default: 16)")
    parser.add_argument("--fp32", action="store_true",
                        help="Use FP32 instead of FP16")
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = Config(
        num_streams=args.streams,
        stream_type=args.type,
        stream_url=args.url,
        model_path=args.model,
        batch_size_min=args.batch_min,
        batch_size_max=args.batch_max,
        use_fp16=not args.fp32,
        duration_seconds=args.duration
    )
    
    # Handle Ctrl+C gracefully
    pipeline = BenchmarkPipeline(config)
    
    def signal_handler(sig, frame):
        pipeline.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    pipeline.run()


if __name__ == "__main__":
    main()

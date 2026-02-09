#!/usr/bin/env python3
"""
Hardware Capacity Benchmark - Test maximum concurrent streams CPU can handle
Uses 1 RTSP source, broadcasts to N workers to avoid camera bottleneck
"""

import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false"

import cv2
import threading
import time
import numpy as np
from collections import deque
from queue import Queue, Empty, Full

# RTSP URL
RTSP_URL = "rtsp://root:Mkvc%402025@192.168.40.40:554/media/stream.sdp?profile=Profile101"
NUM_WORKERS = 200  # Number of simulated streams


class RTSPBroadcaster(threading.Thread):
    """Single RTSP reader that broadcasts frames to multiple workers"""
    
    def __init__(self, rtsp_url: str, output_queues: list):
        super().__init__(daemon=True)
        self.rtsp_url = rtsp_url
        self.output_queues = output_queues
        self.running = False
        self.cap = None
        self.fps = 0.0
        self.frame_count = 0
        self.last_time = time.time()
    
    def run(self):
        self.running = True
        print("📡 Broadcaster: Connecting to RTSP...")
        
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        if not self.cap.isOpened():
            print("❌ Broadcaster: Failed to open RTSP")
            return
        
        print("✅ Broadcaster: Connected, broadcasting to all workers...")
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("⚠️  Broadcaster: Reconnecting...")
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                continue
            
            # Broadcast to all queues (non-blocking)
            for q in self.output_queues:
                try:
                    if q.full():
                        q.get_nowait()  # Drop old frame
                    q.put_nowait(frame.copy())
                except:
                    pass
            
            self.frame_count += 1
            
            # Calculate FPS
            now = time.time()
            if now - self.last_time >= 1.0:
                self.fps = self.frame_count / (now - self.last_time)
                self.frame_count = 0
                self.last_time = now
    
    def stop(self):
        self.running = False
        time.sleep(0.2)
        if self.cap:
            try:
                self.cap.release()
            except:
                pass


class StreamWorker(threading.Thread):
    """Simulated stream worker - processes frames independently"""
    
    def __init__(self, worker_id: int, input_queue: Queue):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.running = False
        self.fps = 0.0
        self.frame_count = 0
        self.last_time = time.time()
        self.latest_frame = None
        self.lock = threading.Lock()
    
    def run(self):
        self.running = True
        
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
            except Empty:
                continue
            
            # Simulate processing (resize, color conversion, etc.)
            processed = cv2.resize(frame, (640, 640))
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            
            # Store latest
            with self.lock:
                self.latest_frame = processed
            
            self.frame_count += 1
            
            # Calculate FPS
            now = time.time()
            if now - self.last_time >= 1.0:
                self.fps = self.frame_count / (now - self.last_time)
                self.frame_count = 0
                self.last_time = now
    
    def get_display_frame(self):
        """Get frame for UI display"""
        with self.lock:
            if self.latest_frame is None:
                return None
            frame = cv2.resize(self.latest_frame, (320, 180))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add overlay
        cv2.putText(frame, f"W{self.worker_id}", (5, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"{self.fps:.1f}FPS", (5, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def stop(self):
        self.running = False


def main():
    print("="*60)
    print(f"🔬 Hardware Capacity Benchmark")
    print(f"📡 1 RTSP source → {NUM_WORKERS} workers")
    print("="*60)
    print()
    
    # Create queues for broadcasting
    queues = [Queue(maxsize=2) for _ in range(NUM_WORKERS)]
    
    # Start broadcaster
    broadcaster = RTSPBroadcaster(RTSP_URL, queues)
    broadcaster.start()
    
    time.sleep(2)  # Wait for connection
    
    # Start workers
    workers = []
    for i in range(NUM_WORKERS):
        worker = StreamWorker(worker_id=i+1, input_queue=queues[i])
        workers.append(worker)
        worker.start()
        time.sleep(0.1)
    
    print()
    print("🖥️  UI Window opened - Press 'q' to quit")
    print()
    
    time.sleep(2)
    
    try:
        while True:
            # Get frames from workers
            frames = []
            for worker in workers:
                frame = worker.get_display_frame()
                if frame is not None:
                    frames.append(frame)
                else:
                    placeholder = np.zeros((180, 320, 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"W{worker.worker_id} Loading...", (50, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    frames.append(placeholder)
            
            if len(frames) < NUM_WORKERS:
                time.sleep(0.1)
                continue
            
            # Create grid (5 columns)
            rows = []
            for i in range(0, NUM_WORKERS, 5):
                row_frames = frames[i:i+5]
                while len(row_frames) < 5:
                    row_frames.append(np.zeros((180, 320, 3), dtype=np.uint8))
                row = np.hstack(row_frames)
                rows.append(row)
            
            grid = np.vstack(rows)
            
            # Add stats
            total_worker_fps = sum(w.fps for w in workers)
            avg_worker_fps = total_worker_fps / NUM_WORKERS if NUM_WORKERS > 0 else 0
            
            info_text = f"Broadcaster: {broadcaster.fps:.1f} FPS | Workers Total: {total_worker_fps:.1f} | Avg: {avg_worker_fps:.1f}"
            cv2.putText(grid, info_text, (10, grid.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow(f"Hardware Capacity Test - {NUM_WORKERS} Workers", grid)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    
    # Shutdown
    print("\n🛑 Stopping...")
    
    broadcaster.stop()
    for worker in workers:
        worker.stop()
    
    broadcaster.join(timeout=3)
    for worker in workers:
        worker.join(timeout=3)
    
    time.sleep(0.5)
    cv2.destroyAllWindows()
    time.sleep(0.3)
    
    # Results
    print()
    print("="*60)
    print("📈 HARDWARE CAPACITY RESULTS")
    print("="*60)
    
    print(f"\nBroadcaster FPS:      {broadcaster.fps:.2f}")
    print(f"\nWorker Performance:")
    
    total_fps = 0
    for worker in workers:
        print(f"  Worker {worker.worker_id:2d}: {worker.fps:5.2f} FPS")
        total_fps += worker.fps
    
    avg_fps = total_fps / NUM_WORKERS if NUM_WORKERS > 0 else 0
    
    print(f"\nTotal Worker FPS:     {total_fps:.2f}")
    print(f"Average per Worker:   {avg_fps:.2f}")
    print()
    
    # Analysis
    efficiency = (avg_fps / broadcaster.fps * 100) if broadcaster.fps > 0 else 0
    
    print("📊 Analysis:")
    print(f"  Efficiency:         {efficiency:.1f}%")
    
    if efficiency > 95:
        print(f"  ✅ EXCELLENT: Hardware can handle {NUM_WORKERS}+ workers")
        print(f"     → Try increasing NUM_WORKERS to find limit")
    elif efficiency > 80:
        print(f"  ✅ GOOD: Hardware handles {NUM_WORKERS} workers well")
    elif efficiency > 60:
        print(f"  ⚠️  MARGINAL: Hardware at ~{NUM_WORKERS} worker limit")
    else:
        print(f"  ❌ OVERLOADED: Reduce NUM_WORKERS to {int(NUM_WORKERS * 0.7)}")
    
    print("="*60)


if __name__ == "__main__":
    main()

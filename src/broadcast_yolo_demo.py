#!/usr/bin/env python3
"""
RTSP Broadcast + Multi-Worker YOLO Inference Demo
1 RTSP source → 5 workers → Real-time person detection with YOLOv11n
"""

import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false"

import cv2
import threading
import time
import numpy as np
from queue import Queue, Empty
from pathlib import Path
from ultralytics import YOLO

# Configuration
RTSP_URL = "rtsp://root:Mkvc%402025@192.168.40.40:554/media/stream.sdp?profile=Profile101"
MODEL_PATH = Path(__file__).parent.parent / "models" / "yolo11n.pt"
NUM_WORKERS = 15
PERSON_CLASS_ID = 0  # COCO class 0 = person
CONFIDENCE_THRESHOLD = 0.5

# Grid configuration (auto-calculated)
GRID_COLS = 6  # 6 columns
GRID_ROWS = (NUM_WORKERS + GRID_COLS - 1) // GRID_COLS  # Auto rows
FRAME_WIDTH = 256  # Smaller for 30 streams
FRAME_HEIGHT = 144


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
        
        print("✅ Broadcaster: Connected!")
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("⚠️  Broadcaster: Reconnecting...")
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                continue
            
            # Broadcast to all queues
            for q in self.output_queues:
                try:
                    if q.full():
                        q.get_nowait()
                    q.put_nowait(frame.copy())
                except:
                    pass
            
            self.frame_count += 1
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


class InferenceWorker(threading.Thread):
    """Worker that runs YOLO inference for person detection"""
    
    def __init__(self, worker_id: int, input_queue: Queue, model_path: str):
        super().__init__(daemon=True)
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.model_path = model_path
        self.running = False
        
        # Stats
        self.fps = 0.0
        self.frame_count = 0
        self.last_time = time.time()
        self.person_count = 0
        self.inference_time_ms = 0.0
        
        # Latest result for display
        self.latest_frame = None
        self.lock = threading.Lock()
        
        # Model (loaded in run)
        self.model = None
    
    def run(self):
        self.running = True
        
        # Load model in worker thread - FORCE GPU
        print(f"🤖 Worker {self.worker_id}: Loading YOLOv11n onto GPU...")
        self.model = YOLO(str(self.model_path), verbose=False)
        self.model.to('cuda:0')
        print(f"✅ Worker {self.worker_id}: Model ready (GPU)!")
        
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
            except Empty:
                continue
            
            # Run inference
            start_time = time.perf_counter()
            results = self.model(frame, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD, verbose=False)
            self.inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Count persons
            self.person_count = 0
            annotated_frame = frame.copy()
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    self.person_count = len(boxes)
                    
                    # Draw boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # Draw green box for person
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Person {conf:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Store result
            with self.lock:
                self.latest_frame = annotated_frame
            
            self.frame_count += 1
            now = time.time()
            if now - self.last_time >= 1.0:
                self.fps = self.frame_count / (now - self.last_time)
                self.frame_count = 0
                self.last_time = now
    
    def get_display_frame(self):
        """Get annotated frame for display"""
        with self.lock:
            if self.latest_frame is None:
                return None
            frame = cv2.resize(self.latest_frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Add compact overlay
        cv2.putText(frame, f"W{self.worker_id}", (3, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, f"{self.fps:.1f}fps", (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        cv2.putText(frame, f"{self.inference_time_ms:.0f}ms", (3, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        
        # Person count with highlight if detected
        color = (0, 255, 0) if self.person_count > 0 else (100, 100, 100)
        cv2.putText(frame, f"P:{self.person_count}", (FRAME_WIDTH - 40, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def stop(self):
        self.running = False


def main():
    print("="*60)
    print("🎯 RTSP Broadcast + 5 Workers YOLO Person Detection")
    print(f"📦 Model: {MODEL_PATH}")
    print("="*60)
    print()
    
    # Create queues
    queues = [Queue(maxsize=2) for _ in range(NUM_WORKERS)]
    
    # Start broadcaster
    broadcaster = RTSPBroadcaster(RTSP_URL, queues)
    broadcaster.start()
    
    time.sleep(2)
    
    # Start workers
    workers = []
    for i in range(NUM_WORKERS):
        worker = InferenceWorker(worker_id=i+1, input_queue=queues[i], model_path=MODEL_PATH)
        workers.append(worker)
        worker.start()
        time.sleep(0.5)  # Stagger model loading
    
    print()
    print("🖥️  Press 'q' to quit")
    print()
    
    time.sleep(3)
    
    try:
        while True:
            # Get frames from workers
            frames = []
            for worker in workers:
                frame = worker.get_display_frame()
                if frame is not None:
                    frames.append(frame)
                else:
                    placeholder = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"W{worker.worker_id} Loading", (20, FRAME_HEIGHT//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    frames.append(placeholder)
            
            if len(frames) < NUM_WORKERS:
                time.sleep(0.1)
                continue
            
            # Create grid (GRID_COLS x GRID_ROWS)
            rows = []
            for row_idx in range(GRID_ROWS):
                start_idx = row_idx * GRID_COLS
                end_idx = min(start_idx + GRID_COLS, NUM_WORKERS)
                row_frames = frames[start_idx:end_idx]
                
                # Pad row if needed
                while len(row_frames) < GRID_COLS:
                    row_frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
                
                row = np.hstack(row_frames)
                rows.append(row)
            
            grid = np.vstack(rows)
            
            # Add stats bar at bottom
            stats_bar = np.zeros((30, grid.shape[1], 3), dtype=np.uint8)
            total_fps = sum(w.fps for w in workers)
            total_persons = sum(w.person_count for w in workers)
            avg_infer = np.mean([w.inference_time_ms for w in workers])
            
            info = f"Broadcaster: {broadcaster.fps:.1f} FPS | Total: {total_fps:.1f} FPS | Avg Infer: {avg_infer:.0f}ms | Persons: {total_persons}"
            cv2.putText(stats_bar, info, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            grid = np.vstack([grid, stats_bar])
            
            cv2.imshow(f"RTSP Broadcast + {NUM_WORKERS} Workers YOLO Detection", grid)
            
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
    
    # Results
    print()
    print("="*60)
    print("📈 FINAL RESULTS")
    print("="*60)
    print(f"Broadcaster FPS:    {broadcaster.fps:.2f}")
    print()
    for worker in workers:
        print(f"Worker {worker.worker_id}: {worker.fps:.2f} FPS | Infer: {worker.inference_time_ms:.1f}ms | Persons: {worker.person_count}")
    print("="*60)


if __name__ == "__main__":
    main()

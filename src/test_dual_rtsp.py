#!/usr/bin/env python3
"""
Test script with UI to check if CPU can handle 15 concurrent RTSP streams
"""

import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false"

import cv2
import threading
import time
import numpy as np
from collections import deque

# RTSP URL
RTSP_URL = "rtsp://root:Mkvc%402025@192.168.40.40:554/media/stream.sdp?profile=Profile101"
NUM_STREAMS = 5

class RTSPStreamTest(threading.Thread):
    """RTSP stream reader with UI support"""
    
    def __init__(self, stream_id: int, rtsp_url: str):
        super().__init__(daemon=True)
        self.stream_id = stream_id
        self.rtsp_url = rtsp_url
        self.running = False
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        self.current_fps = 0.0
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
    
    def run(self):
        self.running = True
        print(f"🎥 Stream {self.stream_id}: Connecting to RTSP...")
        
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        if not self.cap.isOpened():
            print(f"❌ Stream {self.stream_id}: Failed to open RTSP stream")
            return
        
        print(f"✅ Stream {self.stream_id}: Connected successfully")
        
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                print(f"⚠️  Stream {self.stream_id}: Lost connection, reconnecting...")
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                continue
            
            # Store latest frame
            with self.lock:
                self.latest_frame = frame.copy()
            
            self.frame_count += 1
            
            # Calculate FPS every second
            now = time.time()
            if now - self.last_time >= 1.0:
                fps = self.frame_count / (now - self.last_time)
                self.fps_history.append(fps)
                self.current_fps = fps
                self.frame_count = 0
                self.last_time = now
    
    def get_frame(self):
        """Get latest frame with FPS overlay"""
        with self.lock:
            if self.latest_frame is None:
                return None
            
            frame = self.latest_frame.copy()
        
        # Resize to smaller size for grid
        frame = cv2.resize(frame, (320, 180))
        
        # Add FPS overlay
        cv2.putText(frame, f"S{self.stream_id}", (5, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"{self.current_fps:.1f}FPS", (5, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def stop(self):
        self.running = False
        # Give thread time to exit loop
        time.sleep(0.2)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
    
    def get_avg_fps(self):
        if self.fps_history:
            return sum(self.fps_history) / len(self.fps_history)
        return 0.0


def main():
    print("="*60)
    print(f"🧪 Testing CPU capability for {NUM_STREAMS} concurrent RTSP streams")
    print("="*60)
    print()
    
    # Create stream readers
    streams = []
    for i in range(NUM_STREAMS):
        stream = RTSPStreamTest(stream_id=i+1, rtsp_url=RTSP_URL)
        streams.append(stream)
        stream.start()
        time.sleep(0.3)  # Stagger startup to avoid overload
    
    print()
    print("🖥️  UI Window opened - Press 'q' to quit")
    print()
    
    # Wait for first frames
    time.sleep(3)
    
    try:
        while True:
            # Get frames from all streams
            frames = []
            for stream in streams:
                frame = stream.get_frame()
                if frame is not None:
                    frames.append(frame)
                else:
                    # Placeholder for missing frame
                    placeholder = np.zeros((180, 320, 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"S{stream.stream_id} Loading...", (50, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    frames.append(placeholder)
            
            if len(frames) < NUM_STREAMS:
                time.sleep(0.1)
                continue
            
            # Create 5x3 grid
            rows = []
            for i in range(0, NUM_STREAMS, 5):
                row_frames = frames[i:i+5]
                # Pad if needed
                while len(row_frames) < 5:
                    row_frames.append(np.zeros((180, 320, 3), dtype=np.uint8))
                row = np.hstack(row_frames)
                rows.append(row)
            
            grid = np.vstack(rows)
            
            # Add overall stats
            total_fps = sum(s.current_fps for s in streams)
            avg_fps = total_fps / NUM_STREAMS if NUM_STREAMS > 0 else 0
            
            info_text = f"Total FPS: {total_fps:.1f} | Avg per stream: {avg_fps:.1f} | Streams: {NUM_STREAMS}"
            cv2.putText(grid, info_text, (10, grid.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display
            cv2.imshow(f"{NUM_STREAMS} RTSP Streams Test", grid)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    
    # Stop streams
    print("\n🛑 Stopping streams...")
    for stream in streams:
        stream.stop()
    
    # Wait for threads to finish with longer timeout
    for i, stream in enumerate(streams, 1):
        stream.join(timeout=5)
        if stream.is_alive():
            print(f"⚠️  Stream {i} still running after timeout")
    
    # Clear frame references before closing windows
    for stream in streams:
        with stream.lock:
            stream.latest_frame = None
    
    # Small delay before destroying windows
    time.sleep(0.5)
    
    # Close windows
    cv2.destroyAllWindows()
    
    # Final cleanup delay
    time.sleep(0.3)
    
    # Results
    print()
    print("="*60)
    print("📈 TEST RESULTS")
    print("="*60)
    
    total_fps = 0
    for i, stream in enumerate(streams, 1):
        avg_fps = stream.get_avg_fps()
        total_fps += avg_fps
        print(f"Stream {i:2d}: {avg_fps:5.2f} FPS")
    
    avg_per_stream = total_fps / NUM_STREAMS if NUM_STREAMS > 0 else 0
    
    print(f"\nTotal FPS:            {total_fps:.2f}")
    print(f"Average per stream:   {avg_per_stream:.2f}")
    print()
    
    # Verdict
    if avg_per_stream > 15:
        print(f"✅ PASS: CPU can handle {NUM_STREAMS} RTSP streams smoothly (>15 FPS avg)")
    elif avg_per_stream > 10:
        print(f"⚠️  MARGINAL: CPU can handle {NUM_STREAMS} streams but with reduced performance")
    else:
        print(f"❌ FAIL: CPU struggles with {NUM_STREAMS} concurrent RTSP streams")
    
    print("="*60)


if __name__ == "__main__":
    main()

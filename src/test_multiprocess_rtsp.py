#!/usr/bin/env python3
"""
Multiprocessing version - Test CPU with 15 concurrent RTSP streams
Each stream runs in a separate process (no GIL limitation)
"""

import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false"

import cv2
import time
import numpy as np
from multiprocessing import Process, Queue, Value
from collections import deque

# RTSP URL
RTSP_URL = "rtsp://root:Mkvc%402025@192.168.40.40:554/media/stream.sdp?profile=Profile101"
NUM_STREAMS = 15


def rtsp_worker(stream_id: int, rtsp_url: str, frame_queue: Queue, fps_value: Value, running_flag: Value):
    """Worker process to capture RTSP stream"""
    
    print(f"🎥 Process {stream_id}: Starting...")
    
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print(f"❌ Process {stream_id}: Failed to open RTSP")
        return
    
    print(f"✅ Process {stream_id}: Connected")
    
    frame_count = 0
    last_time = time.time()
    
    while running_flag.value:
        ret, frame = cap.read()
        
        if not ret:
            print(f"⚠️  Process {stream_id}: Reconnecting...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            continue
        
        # Resize frame
        frame_small = cv2.resize(frame, (320, 180))
        
        # Add overlay
        cv2.putText(frame_small, f"P{stream_id}", (5, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame_small, f"{fps_value.value:.1f}FPS", (5, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Send frame (non-blocking)
        try:
            if frame_queue.full():
                frame_queue.get_nowait()  # Drop old frame
            frame_queue.put_nowait((stream_id, frame_small))
        except:
            pass
        
        frame_count += 1
        
        # Update FPS
        now = time.time()
        if now - last_time >= 1.0:
            fps = frame_count / (now - last_time)
            fps_value.value = fps
            frame_count = 0
            last_time = now
    
    cap.release()
    print(f"🛑 Process {stream_id}: Stopped")


def main():
    print("="*60)
    print(f"🧪 Multiprocessing Test: {NUM_STREAMS} RTSP streams")
    print("="*60)
    print()
    
    # Create shared resources
    processes = []
    queues = []
    fps_values = []
    running_flags = []
    
    for i in range(NUM_STREAMS):
        q = Queue(maxsize=2)
        fps_val = Value('d', 0.0)
        running = Value('i', 1)
        
        queues.append(q)
        fps_values.append(fps_val)
        running_flags.append(running)
        
        p = Process(target=rtsp_worker, args=(i+1, RTSP_URL, q, fps_val, running))
        processes.append(p)
        p.start()
        time.sleep(0.3)  # Stagger startup
    
    print()
    print("🖥️  UI Window opened - Press 'q' to quit")
    print()
    
    # Wait for first frames
    time.sleep(3)
    
    # Frame storage
    latest_frames = {i+1: None for i in range(NUM_STREAMS)}
    
    try:
        while True:
            # Collect frames from queues
            for i, q in enumerate(queues):
                try:
                    while not q.empty():
                        stream_id, frame = q.get_nowait()
                        latest_frames[stream_id] = frame
                except:
                    pass
            
            # Build grid
            frames_list = []
            for i in range(NUM_STREAMS):
                frame = latest_frames.get(i+1)
                if frame is not None:
                    frames_list.append(frame)
                else:
                    placeholder = np.zeros((180, 320, 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"P{i+1} Loading...", (50, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    frames_list.append(placeholder)
            
            # Create 5x3 grid
            rows = []
            for i in range(0, NUM_STREAMS, 5):
                row_frames = frames_list[i:i+5]
                while len(row_frames) < 5:
                    row_frames.append(np.zeros((180, 320, 3), dtype=np.uint8))
                row = np.hstack(row_frames)
                rows.append(row)
            
            grid = np.vstack(rows)
            
            # Add stats
            total_fps = sum(fps_val.value for fps_val in fps_values)
            avg_fps = total_fps / NUM_STREAMS if NUM_STREAMS > 0 else 0
            
            info_text = f"Total FPS: {total_fps:.1f} | Avg: {avg_fps:.1f} | Processes: {NUM_STREAMS}"
            cv2.putText(grid, info_text, (10, grid.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow(f"{NUM_STREAMS} RTSP Streams (Multiprocessing)", grid)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    # Shutdown
    print("\n🛑 Stopping all processes...")
    
    for flag in running_flags:
        flag.value = 0
    
    for i, p in enumerate(processes, 1):
        p.join(timeout=3)
        if p.is_alive():
            print(f"⚠️  Process {i} still running, terminating...")
            p.terminate()
            p.join(timeout=1)
    
    cv2.destroyAllWindows()
    
    # Results
    print()
    print("="*60)
    print("📈 TEST RESULTS")
    print("="*60)
    
    total_fps = 0
    for i, fps_val in enumerate(fps_values, 1):
        fps = fps_val.value
        total_fps += fps
        print(f"Process {i:2d}: {fps:5.2f} FPS")
    
    avg_fps = total_fps / NUM_STREAMS if NUM_STREAMS > 0 else 0
    
    print(f"\nTotal FPS:          {total_fps:.2f}")
    print(f"Average per stream: {avg_fps:.2f}")
    print()
    
    if avg_fps > 15:
        print(f"✅ PASS: {NUM_STREAMS} streams @ {avg_fps:.1f} FPS avg")
    elif avg_fps > 10:
        print(f"⚠️  MARGINAL: {NUM_STREAMS} streams @ {avg_fps:.1f} FPS avg")
    else:
        print(f"❌ FAIL: {NUM_STREAMS} streams @ {avg_fps:.1f} FPS avg")
    
    print("="*60)


if __name__ == "__main__":
    main()

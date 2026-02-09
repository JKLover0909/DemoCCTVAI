#!/usr/bin/env python3
"""
RTSP Broadcast + 20 Multiprocessing Workers YOLO Inference Demo (Bypassing GIL)
"""

import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false"

import cv2
import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from queue import Empty
from pathlib import Path
from ultralytics import YOLO

# Configuration
RTSP_URL = "rtsp://root:Mkvc%402025@192.168.40.40:554/media/stream.sdp?profile=Profile101"
MODEL_PATH = Path(__file__).parent.parent / "models" / "yolo11n.pt"
NUM_WORKERS = 20
PERSON_CLASS_ID = 0  # COCO class 0 = person
CONFIDENCE_THRESHOLD = 0.5
FRAME_WIDTH = 256
FRAME_HEIGHT = 144
GRID_COLS = 5


def worker_process(worker_id: int, input_queue: mp.Queue, output_queue: mp.Queue, model_path: str, shm_name: str):
    """Worker process: Runs in separate GIL, writes to shared memory"""
    print(f"🤖 Worker {worker_id}: Initializing...")
    
    # Attach to shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    shared_frame = np.ndarray((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8, buffer=shm.buf)
    
    # Load model in process
    try:
        model = YOLO(str(model_path), verbose=False)
        model.to('cuda:0')
        print(f"✅ Worker {worker_id}: GPU Ready")
    except Exception as e:
        print(f"❌ Worker {worker_id}: Failed to load model: {e}")
        return

    fps = 0.0
    frame_count = 0
    last_time = time.time()
    
    while True:
        try:
            frames_obj = input_queue.get(timeout=1.0)
            if frames_obj is None:  # Sentinel
                break
                
            frame = frames_obj
            
            # Run inference
            start = time.perf_counter()
            results = model(frame, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD, verbose=False)
            infer_ms = (time.perf_counter() - start) * 1000
            
            # Post-process
            person_count = 0
            annotated = frame.copy()
            annotated = cv2.resize(annotated, (FRAME_WIDTH, FRAME_HEIGHT))
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    person_count = len(boxes)
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Scale coordinates
                        scale_x = FRAME_WIDTH / frame.shape[1]
                        scale_y = FRAME_HEIGHT / frame.shape[0]
                        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                        
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Add stats
            cv2.putText(annotated, f"W{worker_id}", (3, 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(annotated, f"{fps:.1f}fps", (3, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
            cv2.putText(annotated, f"{infer_ms:.0f}ms", (3, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
            color = (0, 255, 0) if person_count > 0 else (100, 100, 100)
            cv2.putText(annotated, f"P:{person_count}", (FRAME_WIDTH-35, 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Write to shared memory (no serialization!)
            np.copyto(shared_frame, annotated)
            
            # Send only metadata (lightweight)
            try:
                output_queue.put_nowait((worker_id, fps, infer_ms, person_count))
            except:
                pass
            
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now
                
        except Empty:
            continue
        except Exception as e:
            print(f"⚠️ Worker {worker_id} error: {e}")
            time.sleep(0.1)


def main():
    # Set start method to spawn for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    print("="*60)
    print(f"🚀 Multiprocessing YOLO Demo ({NUM_WORKERS} Workers)")
    print("="*60)
    
    # Create queues and shared memory
    input_queues = [mp.Queue(maxsize=2) for _ in range(NUM_WORKERS)]
    output_queue = mp.Queue()
    
    # Create shared memory for each worker (for annotated frames)
    shm_size = FRAME_HEIGHT * FRAME_WIDTH * 3  # uint8 BGR
    shared_memories = []
    shared_arrays = []
    
    print("💾 Creating shared memory buffers...")
    for i in range(NUM_WORKERS):
        shm = shared_memory.SharedMemory(create=True, size=shm_size)
        shared_memories.append(shm)
        # Create numpy view for main process
        arr = np.ndarray((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8, buffer=shm.buf)
        shared_arrays.append(arr)
        # Initialize with black frame
        arr[:] = 0
    
    # Start workers
    processes = []
    print("🔥 Starting workers (this may take a few seconds)...")
    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker_process, 
                       args=(i+1, input_queues[i], output_queue, str(MODEL_PATH), shared_memories[i].name))
        processes.append(p)
        p.start()
        time.sleep(0.1)  # Stagger
    
    # Connect RTSP
    print("📡 Connecting to RTSP...")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("❌ Failed to open RTSP")
        return
        
    print("✅ System Ready!")
    
    # Storage for latest stats (frames are in shared memory)
    latest_stats = {}  # worker_id -> (fps, infer_ms, person_count)
    broadcaster_fps = 0.0
    frame_count = 0
    last_time = time.time()
    
    try:
        while True:
            # 1. Read RTSP
            ret, frame = cap.read()
            if not ret:
                print("⚠️ RTSP Reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
                
            # 2. Broadcast to workers
            for q in input_queues:
                try:
                    if q.full():
                        q.get_nowait()
                    q.put_nowait(frame.copy())
                except:
                    pass
            
            # 3. Collect metadata (non-blocking)
            try:
                while True:
                    worker_id, fps, infer, count = output_queue.get_nowait()
                    latest_stats[worker_id] = (fps, infer, count)
            except Empty:
                pass
            
            # 4. Update Broadcaster FPS
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                broadcaster_fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now
            
            # 5. Render UI (read from shared memory)
            frames_list = []
            total_fps = 0
            total_infer = 0
            total_persons = 0
            workers_active = 0
            
            for i in range(NUM_WORKERS):
                wid = i + 1
                if wid in latest_stats:
                    # Read frame directly from shared memory (zero-copy!)
                    frame_res = shared_arrays[i].copy()
                    w_fps, w_infer, w_count = latest_stats[wid]
                    frames_list.append(frame_res)
                    total_fps += w_fps
                    total_infer += w_infer
                    total_persons += w_count
                    workers_active += 1
                else:
                    placeholder = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"W{wid} Waiting", (20, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    frames_list.append(placeholder)
            
            # Grid layout
            rows = []
            for i in range(0, NUM_WORKERS, GRID_COLS):
                row_frames = frames_list[i:i+GRID_COLS]
                while len(row_frames) < GRID_COLS:
                    row_frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
                rows.append(np.hstack(row_frames))
            grid = np.vstack(rows)
            
            # Stats bar
            stats_h = 40
            stats_bar = np.zeros((stats_h, grid.shape[1], 3), dtype=np.uint8)
            avg_infer = total_infer / workers_active if workers_active > 0 else 0
            
            info = f"Input: {broadcaster_fps:.1f} FPS | Total: {total_fps:.1f} FPS | Avg Infer: {avg_infer:.1f}ms | Persons: {total_persons}"
            cv2.putText(stats_bar, info, (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            final_ui = np.vstack([grid, stats_bar])
            cv2.imshow("Multiprocessing YOLO Broadcast", final_ui)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
        
    print("\n🛑 Shutting down...")
    
    # Broadcast stop signal
    for q in input_queues:
        q.put(None)
        
    for p in processes:
        p.join(timeout=2)
        if p.is_alive():
            p.terminate()
    
    # Cleanup shared memory
    for shm in shared_memories:
        try:
            shm.close()
            shm.unlink()
        except:
            pass
            
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

from ultralytics import YOLO
import cv2
import os
import numpy as np

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0
    return intersection_area / union_area

def filter_overlapping_boxes(boxes, confidences, classes, iou_threshold=0.5):
    """
    Filter overlapping boxes, keeping only the one with highest confidence.
    """
    indices = np.argsort(confidences)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            
        remaining_indices = indices[1:]
        suppress = []
        
        for i, index in enumerate(remaining_indices):
            iou = calculate_iou(boxes[current], boxes[index])
            if iou > iou_threshold:
                suppress.append(i)
        
        indices = np.delete(remaining_indices, suppress)
        
    return keep

def yolo_inference_video(model_path, video_path, output_path="yolo_output.mp4", conf_threshold=0.25, iou_threshold=0.45):
    """
    Run YOLO inference on a video and save the results.
    
    Args:
        model_path (str): Path to YOLO model file (.pt)
        video_path (str): Path to input video file
        output_path (str): Path to save output video
        conf_threshold (float): Confidence threshold for detections (0-1)
        iou_threshold (float): IOU threshold for filtering overlapping boxes
    """
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    # Load YOLO model
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    print(f"✓ Model loaded successfully!")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties:")
    print(f"  - FPS: {fps}")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {total_frames/fps:.2f}s")
    
    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"\nRunning inference with conf>{conf_threshold} and NMS iou>{iou_threshold}")
    print("Processing frames...\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Run YOLO inference on the frame
        # We set iou=1 to disable built-in NMS if we want full manual control, or keep it standard and apply extra filtering
        # Here we just use standard predict
        results = model(frame, conf=conf_threshold, verbose=False)[0]
        
        # Extract boxes
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy()
        
        # Custom filtering for overlapping objects
        keep_indices = filter_overlapping_boxes(boxes, confidences, clss, iou_threshold=iou_threshold)
        
        # Draw filtered boxes
        annotated_frame = frame.copy()
        for idx in keep_indices:
            x1, y1, x2, y2 = map(int, boxes[idx])
            conf = confidences[idx]
            cls = int(clss[idx])
            label = f"{model.names[cls]} {conf:.2f}"
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Write the annotated frame
        out.write(annotated_frame)
        
        frame_count += 1
        
        # Print progress
        if frame_count % 30 == 0:
            print(f"  Frame {frame_count}/{total_frames} - After filter: {len(keep_indices)} objects")
    
    # Release everything
    cap.release()
    out.release()
    
    print(f"\n✓ Inference complete!")
    print(f"✓ Output video saved to: {output_path}")

if __name__ == "__main__":
    # YOLO inference on 3.mp4 using yolo11n.pt model
    model_file = "yolo11n.pt"
    input_video = "3.mp4"
    output_video = "yolo_inference_output.mp4"
    
    # Run inference with filtering
    yolo_inference_video(
        model_path=model_file,
        video_path=input_video,
        output_path=output_video,
        conf_threshold=0.25,
        iou_threshold=0.3  # Objects overlapping more than 30% will be filtered (keep best one)
    )

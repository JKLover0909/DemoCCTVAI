import cv2
import os

def process_video_frames(input_video, output_video="output_video.mp4"):
    """
    Processes all frames from a video and creates a new video.
    
    Args:
        input_video (str): Path to the input video file.
        output_video (str): Path to save the output video.
    """
    if not os.path.exists(input_video):
        print(f"Error: Video file not found at {input_video}")
        return
    
    # Open the input video
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  - FPS: {fps}")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Total frames: {total_frames}")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    
    print("\nProcessing frames...")
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Process frame here (currently just copying)
        # You can add any processing you want here, for example:
        # - frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        # - frame = cv2.GaussianBlur(frame, (5, 5), 0)  # Apply blur
        # - Add text, shapes, filters, etc.
        
        processed_frame = frame  # Replace with your processing
        
        # Write the processed frame
        out.write(processed_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:  # Print progress every 30 frames
            print(f"  Processed {frame_count}/{total_frames} frames...")
    
    # Release everything
    cap.release()
    out.release()
    
    print(f"\n✓ Successfully processed {frame_count} frames")
    print(f"✓ Output video saved to: {output_video}")

if __name__ == "__main__":
    # Process data3.mp4
    input_file = "data3.mp4"
    output_file = "processed_video.mp4"
    
    process_video_frames(input_file, output_file)

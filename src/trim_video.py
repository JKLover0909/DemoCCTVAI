import cv2
import os

def trim_video(input_video, output_video, skip_seconds=10, duration_seconds=10):
    """
    Cuts a video by skipping initial frames and taking a specific duration.
    
    Args:
        input_video (str): Path to the input video file.
        output_video (str): Path to save the trimmed video.
        skip_seconds (int): Number of seconds to skip from the beginning.
        duration_seconds (int): Duration in seconds to extract after skipping.
    """
    if not os.path.exists(input_video):
        print(f"Error: Video file not found at {input_video}")
        return
    
    # Open the input video
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties - AUTO DETECT FPS
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Original video properties:")
    print(f"  - FPS: {fps}")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Duration: {total_frames / fps:.2f}s")
    
    # Calculate frame ranges AFTER getting FPS
    start_frame = skip_seconds * fps  # Frame to start reading from
    end_frame = start_frame + (duration_seconds * fps)  # Frame to stop reading
    
    print(f"\nVideo trimming settings:")
    print(f"  - Skip: {skip_seconds}s ({start_frame} frames)")
    print(f"  - Duration: {duration_seconds}s ({duration_seconds * fps} frames)")
    print(f"  - Frame range: {start_frame} to {end_frame}")
    
    # Check if the requested range is valid
    if end_frame > total_frames:
        print(f"\nWarning: Requested end frame ({end_frame}) exceeds total frames ({total_frames})")
        end_frame = total_frames
        print(f"Adjusted end frame to: {end_frame}")
    
    # Set video position to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    current_frame = start_frame
    
    print(f"\nTrimming video...")
    while current_frame < end_frame:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        out.write(frame)
        
        frame_count += 1
        current_frame += 1
        
        if frame_count % 30 == 0:  # Print progress every 30 frames (1 second)
            print(f"  Processed {frame_count}/{end_frame - start_frame} frames...")
    
    # Release everything
    cap.release()
    out.release()
    
    print(f"\n✓ Successfully trimmed video")
    print(f"✓ Extracted {frame_count} frames ({frame_count / fps:.2f}s)")
    print(f"✓ Output video saved to: {output_video}")

if __name__ == "__main__":
    # Trim data3.mp4: skip first 10s, take next 10s (FPS auto-detected)
    input_file = "data3.mp4"
    output_file = "trimmed_video.mp4"
    
    trim_video(
        input_video=input_file,
        output_video=output_file,
        skip_seconds=10,
        duration_seconds=10
    )

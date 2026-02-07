import cv2
import os

def merge_videos(video1_path, video2_path, output_path="merged_video.mp4"):
    """
    Merges two videos into one by concatenating them.
    
    Args:
        video1_path (str): Path to the first video file.
        video2_path (str): Path to the second video file.
        output_path (str): Path to save the merged video.
    """
    # Check if both videos exist
    if not os.path.exists(video1_path):
        print(f"Error: Video file not found at {video1_path}")
        return
    
    if not os.path.exists(video2_path):
        print(f"Error: Video file not found at {video2_path}")
        return
    
    # Open both videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened():
        print(f"Error: Could not open {video1_path}")
        return
    
    if not cap2.isOpened():
        print(f"Error: Could not open {video2_path}")
        return
    
    # Get properties from first video
    fps1 = int(cap1.get(cv2.CAP_PROP_FPS))
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get properties from second video
    fps2 = int(cap2.get(cv2.CAP_PROP_FPS))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video 1 properties:")
    print(f"  - FPS: {fps1}, Resolution: {width1}x{height1}, Frames: {frames1}, Duration: {frames1/fps1:.2f}s")
    print(f"\nVideo 2 properties:")
    print(f"  - FPS: {fps2}, Resolution: {width2}x{height2}, Frames: {frames2}, Duration: {frames2/fps2:.2f}s")
    
    # Check if videos have compatible properties
    if fps1 != fps2:
        print(f"\nWarning: Videos have different FPS ({fps1} vs {fps2}). Using FPS from first video: {fps1}")
    
    if width1 != width2 or height1 != height2:
        print(f"\nWarning: Videos have different resolutions!")
        print(f"  Video 1: {width1}x{height1}")
        print(f"  Video 2: {width2}x{height2}")
        print(f"  Video 2 will be resized to match Video 1")
    
    # Use properties from first video for output
    fps = fps1
    width = width1
    height = height1
    
    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = 0
    
    # Write all frames from first video
    print(f"\nProcessing Video 1...")
    while True:
        ret, frame = cap1.read()
        if not ret:
            break
        
        out.write(frame)
        total_frames += 1
        
        if total_frames % 30 == 0:
            print(f"  Processed {total_frames}/{frames1} frames...")
    
    print(f"✓ Finished Video 1: {total_frames} frames ({total_frames/fps:.2f}s)")
    
    # Write all frames from second video
    print(f"\nProcessing Video 2...")
    frame_count = 0
    while True:
        ret, frame = cap2.read()
        if not ret:
            break
        
        # Resize if necessary
        if width2 != width or height2 != height:
            frame = cv2.resize(frame, (width, height))
        
        out.write(frame)
        total_frames += 1
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{frames2} frames...")
    
    print(f"✓ Finished Video 2: {frame_count} frames ({frame_count/fps:.2f}s)")
    
    # Release everything
    cap1.release()
    cap2.release()
    out.release()
    
    print(f"\n✓ Successfully merged videos!")
    print(f"✓ Total frames: {total_frames} ({total_frames/fps:.2f}s)")
    print(f"✓ Output video saved to: {output_path}")

if __name__ == "__main__":
    # Merge 1.mp4 and 2.mp4
    video1 = "1.mp4"
    video2 = "2.mp4"
    output = "merged_video.mp4"
    
    merge_videos(video1, video2, output)

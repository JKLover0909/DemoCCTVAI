import cv2
import os

def extract_first_frame(video_path, output_path="first_frame.jpg"):
    """
    Extracts the first frame from a video file and saves it as an image.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the extracted frame.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    ret, frame = cap.read()

    if ret:
        # Save the frame
        cv2.imwrite(output_path, frame)
        print(f"First frame saved successfully to {output_path}")
    else:
        print("Error: Could not read the first frame.")

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    # Example usage with data1.mp4 assuming it's in the same directory
    video_file = "data3.mp4"
    extract_first_frame(video_file)

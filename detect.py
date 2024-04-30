import cv2
from ultralytics import YOLO
from datetime import datetime
import numpy as np
# Load the YOLOv8 model
model = YOLO('car_model/best.pt')

# Open the video file
video_path = "car.mp4"
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
video_name= f"car_output.avi"

# FPS for recording video. Setting 6 as the real fps is 6 to 7
fps = 20
# Video Recorder instance
out = cv2.VideoWriter(video_name,fourcc, fps, (480,848))
# out = cv2.VideoWriter(video_name,fourcc, fps, (1920, 1080))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        annotated_frame=cv2.cvtColor(annotated_frame, cv2.COLOR_BGRA2BGR)
        print(annotated_frame.shape)
        # break
        out.write(annotated_frame)
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
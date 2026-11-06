# Installation command for essential libraries
# pip install opencv-python
# pip install ultralytics

# Import OpenCV (cv2)
# This library allows us to access the webcam, capture video frames, and display images in windows.
import cv2

# Import YOLO from the ultralytics library
# This allows us to load and run YOLO object detection models easily.
from ultralytics import YOLO

# Load the YOLO11s model (pretrained on COCO dataset)
# "yolo11s.pt" → 's' stands for small version, faster but slightly less accurate than larger models
model = YOLO("runs/detect/train2/weights/best.pt")

# 0 = default camera (built-in webcam or first external camera)
# This opens the webcam to start capturing video
cap = cv2.VideoCapture(0)

# Check if the webcam was successfully opened
# If not, print an error message
if not cap.isOpened():
    print("Error: Webcam cannot be opened!")

# Loop continuously → read frames from webcam one by one
while True:
    ret, frame = cap.read()  # ret is True if frame was captured correctly; frame is the image data

    # If the frame is not received properly, break the loop
    if not ret:
        break

    # Run YOLO object detection on the current frame
    results = model(frame)

    # Draw bounding boxes, labels, and confidence scores on the image
    # results[0] is the first (and only) prediction result
    # plot() returns the annotated image
    annotated_frame = results[0].plot()

    # Display the image in a window named "YOLO11s - Webcam Detection..."
    cv2.imshow(" Webcam Detection (Press Q to Quit)", annotated_frame)

    # Check if the 'q' key on the keyboard is pressed
    # If pressed → break the loop and stop webcam detection
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam resource (close the camera)
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
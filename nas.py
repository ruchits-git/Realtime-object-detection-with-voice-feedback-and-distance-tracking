import cv2
import numpy as np
from super_gradients.training import models

# Load YOLO NAS M model pretrained on COCO dataset
yolo_nas_m = models.get("yolo_nas_m", pretrained_weights="coco")

# Function to perform object detection on a frame
def detect_objects(frame, threshold):
    # Perform object detection
    detections = yolo_nas_m.predict(frame)

    # Loop through the detections and draw bounding boxes and labels
    for detection in detections.detections:
        if detection.score > threshold:
            x1, y1, x2, y2 = detection.relative_coordinates[0] * frame.shape[1],
            detection.relative_coordinates[1] * frame.shape[0],
            detection.relative_coordinates[2] * frame.shape[1],
            detection.relative_coordinates[3] * frame.shape[0]

            label = detection.class_name
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open the webcam")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame")
        break

    # Perform object detection on the frame
    frame = detect_objects(frame, 0.5)

    # Display the result
    cv2.imshow('Object Detection', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
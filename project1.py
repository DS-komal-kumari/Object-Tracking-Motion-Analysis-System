

import cv2
import numpy as np
import time
from ultralytics import YOLO
from scipy.spatial import distance

# Load YOLO model
model = YOLO("yolov8n.pt")  # Using YOLOv8 for detection

# Object class for bottle (COCO dataset ID for bottles is 39)
target_class_id = 39  
target_label = "bottle"

# Initialize video capture (webcam or video file)
cap = cv2.VideoCapture(0)  

# Tracking variables
positions = []  # Stores all positions (x, y) of the bottle
start_time = None
end_time = None

# Pixels to real-world meters (adjust based on camera)
pixels_per_meter = 50  # Example: 50 pixels = 1 meter

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Run YOLO object detection
    results = model(frame)

    for result in results:
        for obj in result.boxes.data:
            x1, y1, x2, y2, conf, cls_id = obj.tolist()
            cls_id = int(cls_id)

            if cls_id == target_class_id:  # Detect only the bottle
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # Object center

                if not positions:  # First detection
                    start_time = current_time  

                positions.append((cx, cy))  # Track movement
                end_time = current_time  # Continuously update end time

                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"{target_label} {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Calculate total traveled distance
    total_distance_pixels = sum(
        distance.euclidean(positions[i], positions[i + 1]) for i in range(len(positions) - 1)
    ) if len(positions) > 1 else 0

    total_distance_meters = total_distance_pixels / pixels_per_meter  # Convert to meters

    # Calculate speed
    time_diff = (end_time - start_time) if end_time and start_time else 1
    speed = total_distance_meters / time_diff if time_diff > 0 else 0  # Speed in m/s

    # Draw path of movement
    for i in range(1, len(positions)):
        cv2.line(frame, positions[i - 1], positions[i], (0, 255, 0), 2)

    # Display distance and speed
    cv2.putText(frame, f"Total Distance: {total_distance_meters:.2f} m", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Speed: {speed:.2f} m/s", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Bottle Tracking - Distance & Speed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()


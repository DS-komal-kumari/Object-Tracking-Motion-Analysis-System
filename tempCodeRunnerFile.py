import cv2
import numpy as np
import time

# Open the camera
cap = cv2.VideoCapture(0)

# Background subtractor for detecting motion
fgbg = cv2.createBackgroundSubtractorMOG2()

# Variables to store previous position and time
prev_x, prev_y = None, None
prev_time = None

fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second (default is ~30 FPS)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply background subtraction
    fgmask = fgbg.apply(gray)

    # Find contours (detected objects)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small objects
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Calculate speed
            if prev_x is not None and prev_y is not None:
                current_time = time.time()
                time_diff = current_time - prev_time

                if time_diff > 0:
                    distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                    speed = distance / time_diff  # Pixels per second

                    # Display speed on the frame
                    cv2.putText(frame, f"Speed: {speed:.2f} px/s", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                prev_time = current_time

            prev_x, prev_y = x, y

    # Show the frame
    cv2.imshow('Motion Detection & Speed Calculation', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
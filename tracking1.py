import numpy as np
import cv2

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# Open webcam video stream
cap = cv2.VideoCapture(0)

# Create the output video file
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640, 480))

# Initialize the object tracker
tracker = cv2.TrackerCSRT_create()

# Flag to indicate if an object is being tracked
tracking = False
bbox = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize the frame for faster detection
    frame = cv2.resize(frame, (640, 480))
    # Convert the frame to grayscale for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not tracking:
        # Detect people in the image and get bounding boxes
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

        if len(boxes) > 0:
            bbox = (boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3])
            tracking = tracker.init(frame, bbox)

    if tracking:
        # Update the tracker
        success, bbox = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in bbox]
            # Draw a rectangle around the tracked object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Calculate the position for the blue line in the lower third of the frame
    line_y = int(frame.shape[0] - frame.shape[0] / 3)

    # Draw a blue line in the lower third of the frame
    line_color = (255, 0, 0)  # Blue color (BGR)
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), line_color, 2)

    # Write the output video with the blue line
    out.write(frame.astype('uint8'))

    # Display the resulting frame with annotations
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

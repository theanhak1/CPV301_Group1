import cv2

# Load the pre-trained Haar Cascade Classifier for full body detection
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

people = []  # List to store detected people
next_person_id = 1  # Initialize unique ID counter
frame_count = 0
frame_skip = 2

# Define two lines for counting people
line1_x = 200  # X-coordinate of the first line
line2_x = 400  # X-coordinate of the second line

people_in_count = 0
people_out_count = 0

while True:
    ret, frame = cap.read()
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue
    
    if not ret:
        break

    # Convert the frame to grayscale for body detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform body detection
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Update the list of people and draw bounding boxes with IDs, centroids, and counting lines
    for (x, y, w, h) in bodies:
        person_detected = False
        for person in people:
            px, py, pw, ph, pid = person
            # Check if the detected body is close to an existing person
            if abs(x - px) < 50 and abs(y - py) < 50:
                person[0] = x
                person[1] = y
                person_detected = True
                # Draw a bounding box with the person's ID
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(pid), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Calculate and draw the centroid
                centroid_x = x + w // 2
                centroid_y = y + h // 2
                cv2.circle(frame, (centroid_x, centroid_y), 4, (0, 0, 255), -1)
                # Check if the centroid crosses the counting lines
                if x < line1_x and px >= line1_x:
                    people_out_count += 1
                if x > line2_x and px <= line2_x:
                    people_in_count += 1
                break
        if not person_detected:
            # Assign a new ID to the detected person and add them to the list
            pid = next_person_id
            next_person_id += 1
            people.append([x, y, w, h, pid])
            # Draw a bounding box with the person's ID
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, str(pid), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Calculate and draw the centroid
            centroid_x = x + w // 2
            centroid_y = y + h // 2
            cv2.circle(frame, (centroid_x, centroid_y), 4, (0, 0, 255), -1)

    # Draw the counting lines
    cv2.line(frame, (line1_x, 0), (line1_x, frame.shape[0]), (0, 0, 255), 2)
    cv2.line(frame, (line2_x, 0), (line2_x, frame.shape[0]), (0, 0, 255), 2)

    # Display the frame with person detection and counting
    cv2.imshow("Person Detection and Counting", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("People In: ", people_in_count)
print("People Out: ", people_out_count)

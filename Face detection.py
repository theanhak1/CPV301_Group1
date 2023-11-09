import cv2
import threading
import time

# Khởi tạo CascadeClassifier để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Set the desired frame rate
desired_fps = 25
frame_delay = 1.0 / desired_fps

cv2.namedWindow('Face Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Tracking', 640, 480)

tracking_faces = {}
next_face_id = 1
dist_threshold = 50
tracking_duration = 30

counting_line = [(0, 240), (640, 240)]
enter_count = 0
exit_count = 0

def draw_text(image, text, position, font_scale, color, thickness):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def detect_and_track_faces():
    global next_face_id, enter_count, exit_count

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))

        for face_id in list(tracking_faces.keys()):
            tracker, bbox, _, last_seen_time = tracking_faces[face_id]
            elapsed_time = time.time() - last_seen_time
            if elapsed_time > tracking_duration:
                del tracking_faces[face_id]

        for (x, y, w, h) in faces:
            face_id = None
            centroid_x = x + w // 2
            centroid_y = y + h // 2

            tracked = False
            for fid, (tracker, bbox, _, last_seen_time) in tracking_faces.items():
                dist = ((centroid_x - bbox[0]) ** 2 + (centroid_y - bbox[1]) ** 2) ** 0.5
                if dist < dist_threshold:
                    tracked = True
                    face_id = fid
                    break

            if not tracked:
                bbox = (x, y, w, h)
                tracker = cv2.TrackerCSRT_create()
                tracking_faces[next_face_id] = (tracker, bbox, (centroid_x, centroid_y), time.time())
                face_id = next_face_id
                next_face_id += 1
                tracker.init(frame, bbox)
            else:
                tracking_faces[face_id] = (tracker, bbox, (centroid_x, centroid_y), time.time())

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            draw_text(frame, f"Face {face_id}", (x, y - 5), 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid_x, centroid_y), 4, (0, 0, 255), -1)

            if tracking_faces[face_id][2][1] < counting_line[0][1] and centroid_y >= counting_line[0][1]:
                exit_count += 1
                draw_text(frame, f"Person {face_id} exited", (10, 30), 1, (0, 0, 255), 2)
            elif tracking_faces[face_id][2][1] > counting_line[1][1] and centroid_y <= counting_line[1][1]:
                enter_count += 1
                draw_text(frame, f"Person {face_id} entered", (10, 30), 1, (0, 0, 255), 2)

        cv2.line(frame, counting_line[0], counting_line[1], (255, 0, 0), 2)

        # Hiển thị thông tin trực tiếp trên video
        draw_text(frame, f"Enter: {enter_count}", (10, 70), 1, (0, 0, 255), 2)
        draw_text(frame, f"Exit: {exit_count}", (10, 110), 1, (0, 0, 255), 2)

        cv2.imshow('Face Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

detect_thread = threading.Thread(target=detect_and_track_faces)
detect_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('Face Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detect_thread.join()
cv2.destroyAllWindows()

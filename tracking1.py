import cv2
import numpy as np

# Khởi tạo bộ phát hiện chuyển động
fgbg = cv2.createBackgroundSubtractorMOG2()

# Khởi tạo video capture từ webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi hình ảnh sang grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Loại bỏ nhiễu bằng Gaussian blur
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Cập nhật bộ phát hiện chuyển động
    fgmask = fgbg.apply(gray)

    # Áp dụng một số xử lý để loại bỏ nhiễu và làm nổi bật vùng chuyển động
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, None)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, None)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    # Tìm các đối tượng đã di chuyển
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            (x, y, w, h) = cv2.boundingRect(contour)

            # Loại bỏ các vật thể nhỏ hơn một ngưỡng
            if w > 50 and h > 100:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Hiển thị video
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

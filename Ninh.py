from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Define a function to capture the camera feed
def generate_frames():
    camera = cv2.VideoCapture(0)  # Use the default camera (usually the built-in webcam)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

# Define a route to display the camera feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define a route to render an HTML template that displays the camera feed
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

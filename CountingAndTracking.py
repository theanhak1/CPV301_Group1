import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Initialize the main application window
root = tk.Tk()
root.title("61EB5F34-C8FF-4CFE-934F-534D4934D2D7")

# Create a function to connect to the IP camera
def connect_to_camera():
    camera_url = camera_url_entry.get()
    cap = cv2.VideoCapture(camera_url)

    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                video_label.config(image=photo)
                video_label.photo = photo
                root.update()
            else:
                break
        cap.release()
    else:
        video_label.config(text="Failed to connect to the camera")

# Create an entry field for entering the camera's IP address
camera_url_label = Label(root, text="61EB5F34-C8FF-4CFE-934F-534D4934D2D7:")
camera_url_label.pack()
camera_url_entry = tk.Entry(root)
camera_url_entry.pack()

# Create a button to connect to the camera
connect_button = tk.Button(root, text="Connect", command=connect_to_camera)
connect_button.pack()

# Create a label to display the video feed
video_label = Label(root)
video_label.pack()

# Start the GUI main loop
root.mainloop()




import tkinter as tk
import customtkinter as ctk
import torch
import numpy as np
import cv2
from PIL import Image, ImageTk
import vlc

app = tk.Tk()
app.geometry("600x600")
app.title("Drowsiness Detector")
ctk.set_appearance_mode("dark")

vidFrame = tk.Frame(height=600, width=600)
vidFrame.pack()
vid = ctk.CTkLabel(vidFrame)
vid.pack()

counter = 0
counterLabel = ctk.CTkLabel(
    app,
    text=counter,
    height=40,
    width=120,
    font=("Arial", 20),
    text_color="white",
    fg_color="teal"
)
counterLabel.pack(pady=10)

def reset_counter():
    global counter
    counter = 0

resetButton = ctk.CTkButton(
    app,
    text="Reset Counter",
    command=reset_counter,
    height=40,
    width=120,
    font=("Arial", 20),
    text_color="white",
    fg_color="teal"
)
resetButton.pack()

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp2/weights/last.pt', force_reload=True)

# Open video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Create the MediaPlayer instance outside the detect function

audio_path = "alert.mp3"  # Ensure the file path is correct
p = vlc.MediaPlayer(audio_path)
p.audio_set_volume(100)  # Set volume level (adjust if necessary)

def detect():
    global counter
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        return
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)
    img = np.squeeze(results.render())
    
    if len(results.xywh[0]) > 0:
        dconf = results.xywh[0][0][4]
        dclass = results.xywh[0][0][5]

       # If confidence is high and class is "drowsy" (class 16)
        if dconf.item() > 0.60 and dclass.item() == 16.0:
            if not p.is_playing():  # Ensure it doesn't try to play while already playing
                p.stop()  # Stop any previous playback
                p.play()
            counter += 1

    # Display the video in the tkinter window
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr)
    vid.imgtk = imgtk
    vid.configure(image=imgtk)
    vid.after(10, detect)  # Repeat detection after 10ms
    counterLabel.configure(text=counter)  # Update counter label

detect()
app.mainloop()  
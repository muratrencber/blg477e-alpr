import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sv_ttk
import cv2 as cv
from PIL import Image, ImageTk
import threading
import numpy as np
import csv
from recognizer.lp_recognizer import LPRecognizer
from ultralytics import YOLO
import easyocr
import json

ocr = easyocr.Reader(['en'], gpu=True)
lpr_model = YOLO("alpr_v8n_100ep.pt", verbose=False)
vehicle_detection_model = YOLO("yolov8n.pt", verbose=False)
lp_recognizer = LPRecognizer(lpr_model, vehicle_detection_model, ocr)

current_video_path = None
csv_path = 'output.csv'
stop_event = threading.Event()

def browse_video():
    global current_video_path
    filename = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All Files", "*.*")])
    if filename:
        current_video_path = filename
        video_label.config(text="Selected Video: " + filename)
        update_ui_state('video_selected')

def start_prediction():
    predict_button.config(state='disabled', text="Predicting...")
    stop_event.clear()
    update_ui_state('predicting')
    threading.Thread(target=process_video, args=(current_video_path,), daemon=True).start()

def stop_prediction():
    stop_event.set()

def process_video(video_path):
    vid = cv.VideoCapture(video_path)
    fps = vid.get(cv.CAP_PROP_FPS)
    lp_recognizer.setup_tracking(fps)

    original_width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    original_height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = original_width / original_height

    while not stop_event.is_set():
        ret, original_frame = vid.read()
        if not ret:
            break

        frame_data = lp_recognizer.recognize_frame(original_frame)

        canvas_width = video_canvas.winfo_width()
        canvas_height = video_canvas.winfo_height()
        canvas_aspect_ratio = canvas_width / canvas_height

        if canvas_aspect_ratio > aspect_ratio:
            new_height = canvas_height
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = canvas_width
            new_height = int(new_width / aspect_ratio)

        # Resize the frame for display purposes and place it on a black border frame
        display_frame = cv.resize(original_frame, (new_width, new_height))
        border_frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Calculate the offset to center the image
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2

        # Place the resized image onto the center of the black image
        border_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = display_frame

        # Draw predictions on the border_frame
        draw_predictions(border_frame, frame_data, new_width, new_height, x_offset, y_offset)

        # Convert for Tkinter and update GUI
        border_frame = cv.cvtColor(border_frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(border_frame)
        photo = ImageTk.PhotoImage(image)
        video_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        video_canvas.image = photo
        root.update()

    vid.release()

    if not stop_event.is_set():
        update_ui_state('prediction_complete')
    else:
        update_ui_state('stopped')

def update_ui_state(state):
    if state == 'video_selected':
        predict_button.config(state='normal', text="Predict")
        stop_button.pack_forget()
    elif state == 'predicting':
        browse_button.pack_forget()
        stop_button.pack(side=tk.TOP, pady=(10, 0))
    elif state == 'prediction_complete':
        predict_button.pack_forget()
        predict_button.config(state='normal', text="Predict")
        browse_button.pack(side=tk.TOP, pady=(10, 0))
        predict_button.pack(side=tk.TOP, pady=(10, 0))
        stop_button.pack_forget()
    elif state == 'replaying':
        stop_button.pack(side=tk.TOP, pady=(10, 0))
    elif state == 'stopped':
        browse_button.pack(side=tk.TOP, pady=(10, 0))
        predict_button.pack(side=tk.TOP, pady=(10, 0))
        stop_button.pack_forget()
        predict_button.config(state='normal', text="Predict")

def draw_predictions(border_frame, frame_data, new_width, new_height, x_offset, y_offset):
    for lp_id, lpdata in frame_data.lps.items():
        if lpdata.box:
            lptop, lpbottom = lpdata.box.get_points(new_width, new_height)
            lptop = (lptop[0] + x_offset, lptop[1] + y_offset)
            lpbottom = (lpbottom[0] + x_offset, lpbottom[1] + y_offset)
            cv.rectangle(border_frame, lptop, lpbottom, (255, 0, 0), 3)
            cv.putText(border_frame, lpdata.plate, (lptop[0], lptop[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if lpdata.car_box:
            car_top, car_bottom = lpdata.car_box.get_points(new_width, new_height)
            car_top = (car_top[0] + x_offset, car_top[1] + y_offset)
            car_bottom = (car_bottom[0] + x_offset, car_bottom[1] + y_offset)
            cv.rectangle(border_frame, car_top, car_bottom, (0, 0, 255), 3)

root = tk.Tk()
root.title("License Plate Recognition GUI")
root.geometry("800x600")

main_frame = ttk.Frame(root, padding=10)
main_frame.pack(fill=tk.BOTH, expand=True)

video_canvas = tk.Canvas(main_frame, width=760, height=400)
video_canvas.pack(fill=tk.BOTH, expand=True)

video_label = ttk.Label(main_frame, text="No video selected")
video_label.pack()

stop_button = ttk.Button(main_frame, text="Stop", command=stop_prediction)
stop_button.pack_forget()

browse_button = ttk.Button(main_frame, text="Browse Video", command=browse_video)
browse_button.pack(side=tk.TOP, pady=(10, 0))

predict_button = ttk.Button(main_frame, text="Predict", command=start_prediction, state='disabled')
predict_button.pack(side=tk.TOP, pady=(10, 0))

sv_ttk.set_theme("dark")

root.mainloop()
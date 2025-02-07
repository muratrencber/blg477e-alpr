import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sv_ttk
import cv2 as cv
from PIL import Image, ImageTk
import threading
import numpy as np
from recognizer.lp_recognizer import LPRecognizer
from ultralytics import YOLO
import easyocr
import os
import torch

lpr_model_filename = "alpr_v8n_100ep.pt"
vehicle_detection_model_filename = "yolov8n.pt"

ocr = easyocr.Reader(['en'], gpu=True)
lpr_model = YOLO(lpr_model_filename, verbose=False)
vehicle_detection_model = YOLO(vehicle_detection_model_filename, verbose=False)
lp_recognizer = LPRecognizer(lpr_model, vehicle_detection_model, ocr)

vid = None
current_video_path = None
csv_path = 'output.csv'
stop_event = threading.Event()

current_frame_index = 0

def check_cuda_status():
    if torch.cuda.is_available():
        cuda_status_label.config(text="CUDA ready", image=green_circle)
        gpu_name_label.config(text=torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        cuda_status_label.config(text="CUDA device not found", image=red_circle)
        gpu_name_label.config(text="")

def browse_video():
    global current_video_path
    filename = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All Files", "*.*")])
    if filename:
        current_video_path = filename
        update_file_info(filename)
        update_ui_state('video_selected')

def browse_lpr_model():
    global lpr_model_filename
    filename = filedialog.askopenfilename(filetypes=[("PyTorch model files", "*.pt"), ("All Files", "*.*")])
    if filename:
        lpr_model_filename = filename
        update_model_info()

def browse_vehicle_model():
    global vehicle_detection_model_filename
    filename = filedialog.askopenfilename(filetypes=[("PyTorch model files", "*.pt"), ("All Files", "*.*")])
    if filename:
        vehicle_detection_model_filename = filename
        update_model_info()

def extract_thumbnail(video_path):
    cap = cv.VideoCapture(video_path)
    success, image = cap.read()
    if success:
        height, width, _ = image.shape
        max_size = 100
        if height > width:
            new_height = max_size
            new_width = int(max_size * (width / height))
        else:
            new_width = max_size
            new_height = int(max_size * (height / width))
        resized_image = cv.resize(image, (new_width, new_height))
        image = Image.fromarray(cv.cvtColor(resized_image, cv.COLOR_BGR2RGB))
        cap.release()
        return image
    cap.release()
    return None

def update_file_info(filename):
    video_filename_box.config(state='enabled')
    video_filename_box.delete(0, tk.END)
    video_filename_box.insert(0, os.path.basename(filename))
    video_filename_box.config(state='disabled')
    img = extract_thumbnail(filename)
    if img:
        img = ImageTk.PhotoImage(img)
        thumbnail_label.config(image=img)
        thumbnail_label.image = img

def update_model_info():
    global lpr_model, vehicle_detection_model, lpr_model_filename, vehicle_detection_model_filename, lp_recognizer

    lpr_model = YOLO(lpr_model_filename, verbose=False)
    vehicle_detection_model = YOLO(vehicle_detection_model_filename, verbose=False)
    lp_recognizer = LPRecognizer(lpr_model, vehicle_detection_model, ocr)

    lpr_browse_button.config(text=os.path.basename(lpr_model_filename))
    vehicle_browse_button.config(text=os.path.basename(vehicle_detection_model_filename))


def start_prediction():
    global current_theme, current_frame_index
    current_theme = sv_ttk.get_theme(root)

    current_frame_index = 0
    predict_button.config(state='disabled', text="Predicting...")
    stop_event.clear()
    update_ui_state('predicting')
    threading.Thread(target=process_video, args=(current_video_path,), daemon=True).start()

def stop_prediction():
    stop_event.set()

def process_video(video_path):
    global vid, current_frame_index
    vid = cv.VideoCapture(video_path)
    total_frames = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
    fps = vid.get(cv.CAP_PROP_FPS)
    lp_recognizer.setup_tracking(fps)

    original_width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    original_height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = original_width / original_height

    progress_bar.config(maximum=total_frames)

    while not stop_event.is_set():
        ret, original_frame = vid.read()
        if not ret:
            break

        current_frame_index += 1
        progress_bar.config(value=current_frame_index)

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

        # Resize the frame for display purposes and place it on a border frame
        display_frame = cv.resize(original_frame, (new_width, new_height))
        border_frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        border_frame[:] = (28, 28, 28) if current_theme == 'dark' else (250, 250, 250)

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
        video_canvas_placeholder.pack_forget()
        video_canvas.pack(fill=tk.BOTH, expand=True)
        progress_bar.pack(fill=tk.X)
        stop_button.pack_forget()
    elif state == 'predicting':
        browse_button.config(state='disabled')
        stop_button.pack(side=tk.TOP, pady=(10, 0))
    elif state == 'prediction_complete':
        predict_button.config(state='normal', text="Predict")
        browse_button.config(state='normal')
        stop_button.pack_forget()
    elif state == 'replaying':
        stop_button.pack(side=tk.TOP, pady=(10, 0))
    elif state == 'stopped':
        browse_button.config(state='normal')
        predict_button.pack(side=tk.TOP, pady=(10, 0))
        stop_button.pack_forget()
        predict_button.config(state='normal', text="Predict")

def switch_theme():
    global current_theme
    sv_ttk.toggle_theme()
    current_theme = sv_ttk.get_theme(root)


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

def on_progress_click(event):
    global vid, current_frame_index
    width = progress_bar.winfo_width()
    click_x = event.x 
    progress_ratio = click_x / width
    
    total_frames = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
    target_frame = int(total_frames * progress_ratio)
    
    if vid.isOpened():
        vid.set(cv.CAP_PROP_POS_FRAMES, target_frame)
        progress_bar.config(value=target_frame)
        current_frame_index = target_frame



root = tk.Tk()
root.title("License Plate Recognition GUI")
root.geometry("1000x600")
root.minsize(1000, 600)



main_frame = ttk.Frame(root, padding=10)
main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

sidebar_frame = ttk.Frame(root, padding=10)
sidebar_frame.pack(side=tk.RIGHT, fill=tk.Y)

video_canvas = tk.Canvas(main_frame, width=760, height=400)
video_canvas.pack_forget()

video_canvas_placeholder = ttk.Button(main_frame, text='Select a video to start', command=browse_video)
video_canvas_placeholder.pack(fill=tk.BOTH, expand=True)

progress_bar = ttk.Progressbar(main_frame, orient='horizontal', length=760, mode='determinate')
progress_bar.bind('<Button-1>', on_progress_click)
progress_bar.pack_forget()

predict_button = ttk.Button(sidebar_frame, text="Predict", command=start_prediction, state='disabled', style='Accent.TButton')
predict_button.pack(side=tk.TOP, pady=(10, 0))

file_frame = ttk.LabelFrame(sidebar_frame, text="Video File", padding=10)
file_frame.pack(pady=(10, 0), fill=tk.X)

lpr_model_select_frame = ttk.LabelFrame(sidebar_frame, text="LP Detection Model", padding=10)
lpr_model_select_frame.pack(pady=(10, 0), fill=tk.X)

vehicle_model_select_frame = ttk.LabelFrame(sidebar_frame, text="Vehicle Detection Model", padding=10)
vehicle_model_select_frame.pack(pady=(10, 0), fill=tk.X)

acceleration_frame = ttk.LabelFrame(sidebar_frame, text="Acceleration", padding=10)
acceleration_frame.pack(pady=(10, 0), fill=tk.X)

green_circle = tk.PhotoImage(width=10, height=10)
green_circle.put(("green",), to=(0, 0, 9, 9))

red_circle = tk.PhotoImage(width=10, height=10)
red_circle.put(("red",), to=(0, 0, 9, 9))

cuda_status_label = ttk.Label(acceleration_frame, text="", image=None, compound="left")
cuda_status_label.pack(side=tk.TOP, pady=(5, 0), fill=tk.X)

gpu_name_label = ttk.Label(acceleration_frame, text="")
gpu_name_label.pack(side=tk.TOP, pady=(5, 0), fill=tk.X)

lpr_browse_button = ttk.Button(lpr_model_select_frame, text=lpr_model_filename, command=browse_lpr_model)
lpr_browse_button.pack(pady=(5, 0), fill=tk.X)

vehicle_browse_button = ttk.Button(vehicle_model_select_frame, text=vehicle_detection_model_filename, command=browse_vehicle_model)
vehicle_browse_button.pack(pady=(5, 0), fill=tk.X)

video_filename_box = ttk.Entry(file_frame, state='disabled', width=30)
video_filename_box.pack(side=tk.TOP, fill=tk.X)

thumbnail_label = ttk.Label(file_frame)
thumbnail_label.pack(pady=(5, 0))

browse_button = ttk.Button(file_frame, text="Browse", command=browse_video)
browse_button.pack(pady=(5, 0))

stop_button = ttk.Button(sidebar_frame, text="Stop", command=stop_prediction)
stop_button.pack_forget()

about_label = ttk.Label(sidebar_frame, text="Made in İTÜ with ❤")
about_label.pack(side=tk.BOTTOM, pady=(10, 0))

theme_switch_button = ttk.Checkbutton(sidebar_frame, text="Switch theme", style="Switch.TCheckbutton", command=switch_theme)
theme_switch_button.pack(side=tk.BOTTOM, pady=(5, 0))

sv_ttk.set_theme("dark")
current_theme = sv_ttk.get_theme(root)

check_cuda_status()
root.mainloop()
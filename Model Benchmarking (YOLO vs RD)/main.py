import time # to record how many secs it takes to process the clip
import psutil # to measure the RAM usage consumption
import cv2 # imports Open Computer Vision to open, decode, and extract frames from .mp4 video files.
from ultralytics import YOLO, RTDETR # import the pretrained neural networks

VIDEO_PATH = "2vs1.mp4"

def benchmark_model(model_name, model_class): # accepts the model name (e,g yolo11n.pt) and its class (e.g YOLO)
    print(f"------------BENCHMARKING {model_name}------------")

    model = model_class(model_name) # download and loads the neural network's weights into my laptop memory

    cap = cv2.VideoCapture(VIDEO_PATH) # creates an OpenCV VideoCapture linked to the clip
    frame_count = 0
    start_time = time.time()
    process = psutil.Process() # allows to monitor the exact RAM used from this exact point
    ram_usage_start = process.memory_info().rss / (1024*1024)

    while cap.isOpened(): # loop to continue as long as the video is running
        ret, frame = cap.read() # grabs every next frame of the clip to feed into the AI model, ret is true/false frame is grabbed, frame is raw image matrix (pixels)
        if not ret: # video ended
            break

        results = model(frame, device="cpu", verbose=False) # Core AI calculation, feeds every frame to AI model, run CPU for Design Requirement 7- verbose=False is to be silent
        frame_count += 1

    end_time = time.time()
    cap.release() # close video

    total_time = end_time - start_time
    fps = frame_count / total_time
    ram_usage_end = process.memory_info().rss / (1024 * 1024)

    print(f"Total Frames Processed: {frame_count}")
    print(f"Total Inference Time: {total_time:.2f} seconds")
    print(f"Average FPS: {fps:.2f} FPS")
    print(f"RAM Usage: {ram_usage_end:.2f} MB")
    return fps, total_time, ram_usage_end

yolo_fps, yolo_time, yolo_ram = benchmark_model("yolo11n.pt", YOLO)
rtdetr_fps, rtdetr_time, rtdetr_ram = benchmark_model("rtdetr-l.pt", RTDETR)

from ultralytics import YOLO # Offical library that houses YOLO11 model
import cv2 # Computer vision tool to load video files, draw boxes, save/display output frames
import math
from collections import defaultdict

VIDEO_PATH = "media/clear view sidelines.mp4"

def running_model():
    model = YOLO("yolo11n.pt")

    results = model.track( # Specify "track" for ByteTrack
        source=VIDEO_PATH,
        tracker="/Users/warm/Library/Mobile Documents/com~apple~CloudDocs/Desktop/ML Football/MVP/custom_bytetrack.yaml",
        classes=[0, 32], # specific sports ball & players class IDs
        show=True, 
        save=True,
        stream=True, # Warning sign accepted for RAM usage
        verbose=False # Silences messy Pytorch terminal noise
    )

    frame_number = 0
    track_history = defaultdict(list)

    for result in results:
        boxes = result.boxes
        track_ids = boxes.id.int().tolist() # Unpacks tensor array into native python int elements
        class_ids = boxes.cls.int().tolist() # Captures whether the object is a 0 or a 32

        frame_number += 1
        ball_centre = None # Reset ball tracking every frame

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):

            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()

            if class_id == 32:
                ball_centre = (int((xmin + xmax)/2), int((ymin+ymax)/2))
                continue
            else:
                player_centre = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))

            if ball_centre is not None:
                distance = math.dist(player_centre, ball_centre)
            else:
                distance = -1 # placeholder when ball is not detected

            print(f"Frame {frame_number} -> Player {track_id} at [{int(xmin)}, {int((ymin))}]")
            track_history[track_id].append(
                {
                    "frame": frame_number,
                    "xmin": xmin,
                    "ymin": ymin,
                    "distance_to_ball": distance
                }
            )

    print("------------------------------------------------------------------")
    total_records = 0
    broken_records = 0

    for player_id, frames in track_history.items():
        for f in frames:
            total_records += 1
            if f["distance_to_ball"] == -1:
                broken_records += 1
                print(f"Validation Failure on Frame {f['frame']}: Failed to calculate distance to ball for Player {player_id}.")

    print("------------------------------------------------------------------")
    print(f"Total Player Tracking Instances: {total_records}")
    print(f"Total Broken Distance Calculations: {broken_records}")
    if total_records > 0:
        print(f"Error Rate: {(broken_records / total_records) * 100:.2f}%")
    print("------------------------------------------------------------------")
            
running_model()
from ultralytics import YOLO # Offical library that contains YOLO11 model
import cv2 # Computer vision tool to load video files, draw boxes, save/display output frames
import math
from collections import defaultdict

VIDEO_PATH = "media/clear view sidelines.mp4"

def running_model():
    model = YOLO("yolo11n.pt")

    results = model.track(
        source=VIDEO_PATH,
        tracker="/Users/warm/Library/Mobile Documents/com~apple~CloudDocs/Desktop/ML Football/MVP/custom_bytetrack.yaml",
        classes=[0, 32], # specific sports ball & players class IDs
        show=True, 
        save=True,
        persist=True, # Continous ID for each frame as much as it can
        stream=True, # Warning sign accepted for RAM usage
        verbose=False # Silences messy Pytorch terminal noise
    )

    # {track_id: [(x,y), (x, y), ...]}
    ball_history = defaultdict(list)
    player_history = defaultdict(list) 
    frame_count = 0

    for result in results:
        boxes = result.boxes.xywh.cpu().numpy() # Extract coords as centre-x, centre-y, width, heigth
        track_ids = result.boxes.id.int().cpu().tolist() # These are the persistent IDs
        class_ids = result.boxes.cls.int().cpu().tolist()

        current_players = []
        current_ball = []
        frame_count += 1

        # Next, I need to record and process coords line-by-line
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x, y, w, h = box

            # Recording the centre coord of the object
            centre_coord = (float(x), float(y))

            if class_id == 32:
                ball_history[track_id].append(centre_coord)
                current_ball.append(f"Ball ID {track_id} at {centre_coord}")
            else:
                player_history[track_id].append(centre_coord)
                current_players.append(f"ID {track_id} ({centre_coord})")

        print(f"---------------FRAME {frame_count}---------------")
        if current_ball:
            print(f"Ball Detected: {', '.join(current_ball)}")
        else:
            print(f"Ball Detected: None")
        print(f"Players on screen ({len(current_players)}):")
        # Grouping player coords in rows of 4
        for i in range(0, len(current_players), 4):
            print("   " + ", ".join(current_players[i:i+4]))
        print(f"-------------------------------------------------")

    print("\n" + "="*40)
    print("TRACKING COMPLETED")
    print(f"Total frames: {frame_count}")
    print(f"Total unique players logged: {len(player_history)}")
    print(f"Total unique balls logged: {len(ball_history)}")
    print("="*40)



# print("------------------------------------------------------------------")
# total_records = 0
# broken_records = 0

# for player_id, frames in track_history.items():
#     for f in frames:
#         total_records += 1
#         if f["distance_to_ball"] == -1:
#             broken_records += 1
#             print(f"Validation Failure on Frame {f['frame']}: Failed to calculate distance to ball for Player {player_id}.")

# print("------------------------------------------------------------------")
# print(f"Total Player Tracking Instances: {total_records}")
# print(f"Total Broken Distance Calculations: {broken_records}")
# if total_records > 0:
#     print(f"Error Rate: {(broken_records / total_records) * 100:.2f}%")
# print("------------------------------------------------------------------")
            
running_model()
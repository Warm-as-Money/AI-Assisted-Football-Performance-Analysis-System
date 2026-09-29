from ultralytics import YOLO # Offical library that contains YOLO11 model
import cv2 # Computer vision tool to load video files, draw boxes, save/display output frames
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from collections import defaultdict

VIDEO_PATH = "media/clear view sidelines.mp4"

def running_model():
    model = YOLO("yolo11n.pt")

    results = model.track(
        source=VIDEO_PATH,
        tracker="/Users/warm/Library/Mobile Documents/com~apple~CloudDocs/Desktop/ML Football/MVP/custom_bytetrack.yaml",
        classes=[0, 32], # specific sports ball & players class IDs
        show=False, 
        save=True,
        persist=True, # Continous ID for each frame as much as it can
        stream=True, # Warning sign accepted for RAM usage
        verbose=False # Silences messy Pytorch terminal noise
    )

    # {track_id: [(frame1,x,y), (frame2,x, y), ...]}
    ball_history = defaultdict(list)
    player_history = defaultdict(list) 
    frame_count = 0

    # Pass 1
    for result in results:
        frame_count += 1
        boxes = result.boxes.xywh.cpu().numpy() # Extract coords as centre-x, centre-y, width, heigth
        track_ids = result.boxes.id.int().cpu().tolist() # These are the persistent IDs
        class_ids = result.boxes.cls.int().cpu().tolist()

        current_players = []
        current_ball = []

        # Next, I need to record and process coords line-by-line
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x, y, w, h = box

            # Recording the centre coord of the object
            frame_centre_coord = (frame_count, float(x), float(y))

            if class_id == 32:
                ball_history[track_id].append(frame_centre_coord)
                current_ball.append(f"Ball ID {track_id} at {frame_centre_coord}")
            else:
                player_history[track_id].append(frame_centre_coord)
                current_players.append(f"ID {track_id} ({frame_centre_coord})")

        print(f"---------------FRAME {frame_count}---------------")
        if current_ball:
            print(f"Ball Detected: {', '.join(current_ball)}")
        else:
            print(f"Ball Detected: None")
        print(f"Players on screen ({len(current_players)}):")
        # Grouping player coords in rows of 4
        row = []
        for player in current_players:
            row.append(player)
            if len(row) == 4:
                print("   " + ", ".join(row))
                row = []
        if row:
            print("  "+", ".join(row))
        print(f"-------------------------------------------------")

    print("\n" + "="*40)
    print("TRACKING COMPLETED")
    print(f"Total frames: {frame_count}")
    print(f"Total unique players logged: {len(player_history)}")
    print(f"Total unique balls logged: {len(ball_history)}")
    print("="*40)

    # Pass 2
    # Linear interpolation of the ball data
    all_ball_detecttions = []
    interpolated_ball_history = {}

    for track_id, trajectory in ball_history.items():
        for frame, x, y, in trajectory:
            all_ball_detecttions.append((frame, x, y))

        all_ball_detecttions.sort(key=lambda pt: pt[0])

        frames = np.array([pt[0] for pt in all_ball_detecttions])
        xs = np.array([pt[1] for pt in all_ball_detecttions])
        ys = np.array([pt[2] for pt in all_ball_detecttions])

        f_x = interp1d(frames, xs, kind="linear", fill_value="extrapolate")
        f_y = interp1d(frames, ys, kind="linear", fill_value="extrapolate")

        start_frame = int(frames.min())
        end_frame = int(frames.max())
        all_frames = np.arange(start_frame, end_frame + 1)

        interpolated_xs = f_x(all_frames)
        interpolated_ys = f_y(all_frames)

        filled_trajectory = {
            int(frame): (float(cx), float(cy))
            for frame, cx, cy in zip(all_frames, interpolated_xs, interpolated_ys)
        }
        interpolated_ball_history[track_id] = filled_trajectory

    return interpolated_ball_history, frame_count

def get_ball_position(frame_num, ball_history_data):    
    for track_id, trajectory in ball_history_data.items():        
        if frame_num in trajectory:            
            return trajectory[frame_num]    
    return None

interpolated_ball_history, frame_count = running_model()

covered = 0
missing = 0
for f in range(1, frame_count+1):
    position = get_ball_position(f, interpolated_ball_history)
    if position is not None:
        covered += 1
    else:
        missing += 1

print(f"frames with ball positions: {covered}")
print(f"frames without ball positions: {missing}")


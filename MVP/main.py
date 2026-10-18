from ultralytics import YOLO
import cv2
import math
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
        show=True, 
        save=True,
        persist=True, # Continous ID for each frame as much as it can
        stream=True, # Warning sign accepted for RAM usage
        verbose=False # Silences messy Pytorch terminal noise
    )

    # {track_id: [(frame1,x,y), (frame2,x, y), ...]}
    ball_history = defaultdict(list)
    player_history = defaultdict(list) 
    all_frame_boxes = defaultdict(dict) #{frame_count: {track_id: (x, y, w, h)}}
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

            if class_id != 32:
                all_frame_boxes[frame_count][track_id] = (float(x), float(y), float(w), float(h))

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
    for track_id, trajectory in ball_history.items():
        for frame, x, y, in trajectory:
            all_ball_detecttions.append((frame, x, y))

    # Sort chronological order
    all_ball_detecttions.sort(key=lambda pt: pt[0])

    interpolated_ball_history = {}

    if len(all_ball_detecttions) > 0:
        #starting and ending boundaries
        start_frame = int(all_ball_detecttions[0][0])
        end_frame = int(all_ball_detecttions[-1][0])
        
        ball_lookup = {int(pt[0]): (float(pt[1]), float(pt[2])) for pt in all_ball_detecttions}
        
        for f in range(start_frame, end_frame + 1):
            if f in ball_lookup:
                # If the ball is present
                interpolated_ball_history[f] = ball_lookup[f]
            else:
                # If the ball is missing, find the nearest previous frame with data
                prev_f = f - 1
                while prev_f not in ball_lookup and prev_f >= start_frame:
                    prev_f -= 1

                #upcoming frame with data 
                next_f = f + 1
                while next_f not in ball_lookup and next_f <= end_frame:
                    next_f += 1
                    
                # rise over run
                if prev_f in ball_lookup and next_f in ball_lookup:
                    x1, y1 = ball_lookup[prev_f]
                    x2, y2 = ball_lookup[next_f]
                    
                    gap_percentage = (f - prev_f) / (next_f - prev_f)
                    new_x = x1 + gap_percentage * (x2 - x1)
                    new_y = y1 + gap_percentage * (y2 - y1)
                    
                    interpolated_ball_history[f] = (new_x, new_y)

        return interpolated_ball_history, player_history, frame_count, all_frame_boxes, VIDEO_PATH

def get_ball_position(frame_num, ball_history_data):    
    return ball_history_data.get(frame_num, None)

interpolated_ball_history, player_history, frame_count, all_frame_boxes, VIDEO_PATH = running_model()

# Player Identification
def draw_player_frame_boxes(player_history, all_frame_boxes, VIDEO_PATH):
    cap = cv2.VideoCapture(VIDEO_PATH)
    current_frame = 1
    player_identified = False

    while not player_identified:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)
        ret, frame = cap.read()
        if ret:
            display_img = frame.copy()
            for track_id, (x,y,w,h) in all_frame_boxes[current_frame].items():
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_img, f"ID {track_id}", (x1, y1 -10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Navigate frames - A/D to move, SPACE to select, Q to quit", display_img)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('d'):
                current_frame = min(current_frame + 1, frame_count)
            elif key == ord('a'):
                current_frame = max(current_frame - 1, 1)
            elif key == ord(' '):
                player_identified = True
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None
        else:
            print(f"ERROR: Frame {frame_count} cannot be identified \n") 

    cap.release()
    cv2.destroyAllWindows()
    return current_frame

selected_frame = draw_player_frame_boxes(player_history, all_frame_boxes, VIDEO_PATH)
print(f"User selected frame: {selected_frame}")

def select_player_box(selected_frame, all_frame_boxes, VIDEO_PATH):
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame - 1)
    ret, frame = cap.read()
    cap.release()

    display_img = frame.copy()
    for track_id, (x,y,w,h) in all_frame_boxes[selected_frame].items():
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_img, f"ID {track_id}", (x1, y1 -10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)


    global mouse_x, mouse_y, click_triggered
    mouse_x = -1
    mouse_y = -1
    click_triggered = False

    def mouse_click(event, x, y, flags, param):
        global mouse_x, mouse_y, click_triggered
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_x = x
            mouse_y = y
            click_triggered = True

    cv2.imshow("Click on yourself to receive feedback", display_img)
    cv2.setMouseCallback("Click on yourself to receive feedback", mouse_click)

    selected_player_id = None
    while True:
        cv2.imshow("Click on yourself to receive feedback", display_img)
        key = cv2.waitKey(30) & 0xFF
        
        if click_triggered:
            for track_id, (cx, cy, w, h) in all_frame_boxes[selected_frame].items():
                x1 = int(cx - w/2)
                y1 = int(cy - h/2)
                x2 = int(cx + w/2)
                y2 = int(cy + h/2)
            
                if mouse_x >= x1 and mouse_x <= x2:
                    if mouse_y >= y1 and mouse_y <= y2:
                        selected_player_id = track_id
                        print(f"Player {track_id} selected as main character")
                        break
            
            if selected_player_id is not None:
                break
            click_triggered = False

    cv2.destroyAllWindows()
    return selected_player_id


main_player_identified = select_player_box(selected_frame, all_frame_boxes, VIDEO_PATH)
print(f"Main player seed ID: {main_player_identified}")

# Checkpoint to see every frame with the ball is covered (interpolation success)
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

# Player to ball distances
def get_player_to_ball_distances(player_history, interpolated_ball_history, frame_count):
    player_to_ball_master_dict = defaultdict(list)
    for player_id, player_detection in player_history.items():
        for frame_num, x_player, y_player in player_detection:
            ball_pos = get_ball_position(frame_num, interpolated_ball_history)
            x_ball, y_ball = ball_pos
            # Calculating the distance between player_pos and ball_pos in the same frame
            player_to_ball_distance = math.sqrt((x_ball - x_player)**2 + (y_ball - y_player)**2)

            player_to_ball_master_dict[player_id].append({
                "frame": frame_num,
                "player_coord": (x_player, y_player),
                "ball_coord": (x_ball, y_ball),
                "player_to_ball_distance": player_to_ball_distance
            })
    return player_to_ball_master_dict



player_ball_distances = get_player_to_ball_distances(player_history, interpolated_ball_history, frame_count)

# The Spatiotemporal Stitching Engine
def stitching_engine(main_player_identified, player_history, player_ball_distances):
    unified_identity = [main_player_identified]
    current_id = main_player_identified

    while True:
        current_last_frame, current_x, current_y = player_history[current_id][-1]
        candidates = []
        for candidate_id, history in player_history.items():
            if not history:
                continue
            first_frame, first_x, first_y = history[0]

            # Condition: first appearance frame > current_id's last frame and not already in unified_identity
            if first_frame > current_last_frame and candidate_id not in unified_identity:
                # Δt = candidate first frame - current last frame
                dt = first_frame - current_last_frame
                # Δd = Euclidean distance between positions
                dd = math.sqrt((first_x - current_x) ** 2 + (first_y - current_y) ** 2)

                # From here, I need to filter to only keep candidates where Δt < 30 and Δd < 200
                # Calculate avg_ball = average ball distance in first 10 frames
                if dt < 60 and dd < 300:
                    ball_records = player_ball_distances.get(candidate_id, [])[:10]
                    if ball_records:
                        total_ball_dist = sum(record["player_to_ball_distance"] for record in ball_records)
                        avg_ball = total_ball_dist / len(ball_records)
                    else:
                        avg_ball = float('inf') # Safety net if data is missing
                        
                    candidates.append((candidate_id, avg_ball))
                
        # acndidates are found, then pick the one with the LOWEST avg_ball
        if candidates:
            #avg_ball ascending
            candidates.sort(key=lambda x: x[1])
            best_candidate_id, _ = candidates[0]
            
            unified_identity.append(best_candidate_id)
            current_id = best_candidate_id
        else:
            break
    return unified_identity
unified_identity = stitching_engine(main_player_identified, player_history, player_ball_distances)
print(f"Unified identity chain: {unified_identity}")

def unifying(unified_identity, player_history):
    unified_player_profile = []
    
    for fragment_id in unified_identity:
        detections = player_ball_distances.get(fragment_id, [])
        unified_player_profile.extend(detections)
        
    unified_player_profile.sort(key=lambda x: x["frame"])
    
    return unified_player_profile

unified_player_profile = unifying(unified_identity, player_ball_distances)
print(f"Total unified detections: {len(unified_player_profile)}")
print(f"First detection: {unified_player_profile[0]}")
print(f"Last detection: {unified_player_profile[-1]}")


attacking_direction = input("Which direction is your team attacking? (left/right): ").strip().lower()
playing_position = input("What position are you playing? (attacker/midfielder/defender): ").strip().lower()



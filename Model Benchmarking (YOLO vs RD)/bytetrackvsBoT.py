import time, os, psutil
from ultralytics import YOLO

VIDEO_PATH = "test_input.mp4"

def run_tracker_benchmark(model_name, config_file):
    print(f"------------BENCHMARKING Tracker: {model_name}, {config_file} ------------")

    # Process to monitor RAM
    process = psutil.Process(os.getpid())
    inital_memory = process.memory_info().rss / (1024 * 1024) # in MB
    model = YOLO("yolo11n.pt")

    # Track unique IDs
    seen_ids = set()
    frame_count = 0
    peak_memory = inital_memory
    start_time = time.time()

    results = model.track(
        source=VIDEO_PATH,
        tracker=config_file,
        classes=[0, 32],
        stream=True,
        show=False,
        save=False
    ) 

    for result in results:
        frame_count += 1

        # Track unique IDs assigned by the tracking engine
        if result.boxes is not None and result.boxes.id is not None:
            # Convert PyTorch Ids to a Python list
            tracker_ids = result.boxes.id.int().tolist()
            seen_ids.update(tracker_ids)

        # Sample peak RAM usage during execution
        current_memory = process.memory_info().rss / (1024 * 1024)
        if current_memory > peak_memory:
            peak_memory = current_memory

    end_time= time.time()
    total_execution_time = end_time - start_time
    fps = frame_count / total_execution_time if total_execution_time > 0 else 0
    net_peak_ram = peak_memory - inital_memory
    total_unique_identities = len(seen_ids)
    
    print(f"Total Frames Processed: {frame_count}")
    print(f"Total Execution Time:  {total_execution_time:.2f} seconds")
    print(f"Throughput Speed:      {fps:.2f} FPS")
    print(f"Net Peak RAM Added:    {net_peak_ram:.2f} MB")
    print(f"Total Unique IDs Found: {total_unique_identities}")

bytetrack_data = run_tracker_benchmark("ByteTrack", "bytetrack.yaml")
botsort_data = run_tracker_benchmark("BoT-SORT", "botsort.yaml")
from pathlib import Path
import pickle
import os

import cv2
import numpy as np
import torch

# BOXMOT_RESUME_PATH = 'boxmot-resume-tracks'
# import sys
# sys.path.insert(0, BOXMOT_RESUME_PATH)
from boxmot import BotSort

from ultralytics import YOLO

import torch
import numpy as np
import random
# 1. Set a global seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 2. Enforce deterministic algorithms
torch.use_deterministic_algorithms(True)


device = torch.device('cuda')

# Initialize the tracker
tracker = BotSort(
    reid_weights=Path('clip_market1501.pt'),
    device=0,
    half=False,
    track_buffer=100
)

# Initialize the detector
detector = YOLO('yolo12x.pt')

# Open the video file
vid = cv2.VideoCapture("car-2.mp4")

# Get video properties for output
fps = int(vid.get(cv2.CAP_PROP_FPS))
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('out-3.mp4', fourcc, fps, (width, height))

# Set variables
READ_EVERY_N_FRAMES = 500
frame_count = -1  # Start at -1 to increment before processing the first frame

# Storage files
STORAGE_DIR = 'tracking_storage'
DETS_FILE = os.path.join(STORAGE_DIR, 'last_dets.pkl')
FRAME_FILE = os.path.join(STORAGE_DIR, 'last_frame.jpg')

# Create storage directory
os.makedirs(STORAGE_DIR, exist_ok=True)

# Initialize storage variables
last_tracks = np.empty((0, 7), dtype=np.float32)

def save_detection_data(tracks, frame, frame_width, frame_height):
    """Save tracks as detections with manual IDs and frame to local storage"""
    # Convert tracks to detections with manual IDs
    # Track format: [x1, y1, x2, y2, track_id, conf, cls, det_ind]
    # Detection format: [x1, y1, x2, y2, conf, cls, manual_id]
    
    if len(tracks) > 0:
        # Normalize coordinates to 0-1 range
        normalized_dets = np.zeros((len(tracks), 8), dtype=np.float32)
        normalized_dets[:, 0] = tracks[:, 0] / frame_width   # x1
        normalized_dets[:, 1] = tracks[:, 1] / frame_height  # y1
        normalized_dets[:, 2] = tracks[:, 2] / frame_width   # x2
        normalized_dets[:, 3] = tracks[:, 3] / frame_height  # y2
        normalized_dets[:, 4] = tracks[:, 5]  # conf
        normalized_dets[:, 5] = tracks[:, 6]  # cls
        normalized_dets[:, 6] = tracks[:, 4]  # manual_id = track_id
    else:
        normalized_dets = np.empty((0, 7), dtype=np.float32)
    
    # Save normalized detections with frame dimensions
    detection_data = {
        'detections': normalized_dets,
        'original_width': frame_width,
        'original_height': frame_height
    }
    
    with open(DETS_FILE, 'wb') as f:
        pickle.dump(detection_data, f)
    
    # Save frame
    cv2.imwrite(FRAME_FILE, frame)
    
def load_last_detections_for_init(current_width, current_height):
    """Load last detections and extract dets and manual IDs separately"""
    try:
        with open(DETS_FILE, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            normalized_dets = data['detections']
        else:
            return None, None
        
        if len(normalized_dets) > 0:
            # Scale to current resolution
            scaled_dets = normalized_dets.copy()
            scaled_dets[:, 0] *= current_width   # x1
            scaled_dets[:, 1] *= current_height  # y1
            scaled_dets[:, 2] *= current_width   # x2
            scaled_dets[:, 3] *= current_height  # y2
            
            # Extract detections (first 6 columns) and manual IDs (last column)
            dets = scaled_dets[:, :6]  # x1,y1,x2,y2,conf,cls
            manual_ids = scaled_dets[:, 6].astype(int)  # manual_id
            
            return dets, manual_ids
        else:
            return np.empty((0, 6), dtype=np.float32), np.array([], dtype=int)
            
    except FileNotFoundError:
        return None, None

def load_last_frame(target_width=None, target_height=None):
    """Load last frame from storage and optionally resize to target resolution"""
    try:
        if os.path.exists(FRAME_FILE):
            frame = cv2.imread(FRAME_FILE)
            if frame is not None and target_width is not None and target_height is not None:
                frame = cv2.resize(frame, (target_width, target_height))
            return frame
        return None
    except:
        return None

def draw_tracks_on_frame(frame, tracks):
    """Draw bounding boxes and IDs on frame"""
    frame_copy = frame.copy()
    for track in tracks:
        if len(track) >= 5:
            x1, y1, x2, y2, track_id = track[:5]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw track ID
            label = f'ID: {track_id}'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return frame_copy

# Check if there are stored detections and frame to resume from
last_dets, last_manual_ids = load_last_detections_for_init(width, height)
last_frame = load_last_frame(width, height)

if last_dets is not None and last_frame is not None and len(last_dets) > 0:
    print("Found previous detection data. Initializing tracker state...")
    print(f"Initializing with {len(last_dets)} detections using manual IDs: {last_manual_ids}")
    
    # Initialize tracker with previous state using the new clean method
    tracker.initialize_from_detections(last_dets, last_frame, last_manual_ids)
else:
    print("No previous detection data found. Starting fresh...")

while True:
    # Capture frame-by-frame
    ret, frame = vid.read()

    if not ret:
        break
    
    frame_count += 1
    
    # Check if we should process this frame for detection
    if frame_count % READ_EVERY_N_FRAMES == 0:
        print(f"Processing frame {frame_count} for detection...")
        
        # Detect objects in the frame
        results = detector.predict(frame, device=device, verbose=True, classes=[2], conf=0.1)
        
        # Process results for tracking
        dets = []
        for r in results:
            if len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy().reshape(-1, 1)
                cls = r.boxes.cls.cpu().numpy().reshape(-1, 1)
                dets.append(np.hstack([xyxy, conf, cls]))

        if dets:
            dets = np.vstack(dets).astype(np.float32)
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        # Update the tracker
        res = tracker.update(dets, frame)   # --> M X (x, y, x, y, id, conf, cls, ind)
        
        # Save tracks as detections with manual IDs for next resume
        save_detection_data(res, frame, width, height)
        last_tracks = res.copy() if len(res) > 0 else np.empty((0, 8), dtype=np.float32)
        
        # Draw tracks on frame
        frame_with_tracks = draw_tracks_on_frame(frame, res)
        
    else:
        # For non-detection frames, just draw the last tracks
        frame_with_tracks = draw_tracks_on_frame(frame, last_tracks)
    
    # Write frame to output video
    out.write(frame_with_tracks)

# Release resources
vid.release()
out.release()

print(f"Output video saved as 'out-2.mp4'")
print(f"Tracking data stored in '{STORAGE_DIR}' directory")
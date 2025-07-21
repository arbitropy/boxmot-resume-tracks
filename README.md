# Fork Description
This fork of boxmot adds the ability to resume object tracking from a previous state.
It does this by taking detections and their known IDs from a prior tracking session.
The primary goal is to maintain persistent IDs for tracked objects across process restarts.
This functionality is especially useful for tracking relatively stationary objects.

## Example Usage

### Storage
The system automatically saves tracking data every `READ_EVERY_N_FRAMES`:
- **Detections**: Normalized bounding boxes with track IDs stored as `last_dets.pkl`
- **Frame**: Last processed frame saved as `last_frame.jpg`
- **Location**: All data stored in `tracking_storage` directory

### Resume Process
When restarting, the system:
1. Loads previous detections and frame from storage
2. Initializes tracker state using `initialize_from_detections()` method
3. Continues tracking with preserved IDs

## Usage

### Basic Setup
````python
# Configure storage
STORAGE_DIR = 'tracking_storage'
READ_EVERY_N_FRAMES = 500  # Save state every N frames

# Initialize tracker
tracker = BotSort(
    reid_weights=Path('clip_market1501.pt'),
    device=0,
    half=False,
    track_buffer=100
)
````

### Storage Functions
````python
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
        return Nonedef load_last_detections_for_init(current_width, current_height):
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
````

### Resume Initialization
````python
# Check for existing tracking data
last_dets, last_manual_ids = load_last_detections_for_init(width, height)
last_frame = load_last_frame(width, height)

if last_dets is not None and len(last_dets) > 0:
    # Resume from previous state
    tracker.initialize_from_detections(last_dets, last_frame, last_manual_ids)
else:
    # Start fresh tracking
    print("Starting fresh tracking...")
````

## Key Features

- **Automatic State Saving**: Tracks are saved automatically at specified intervals
- **Resolution Independence**: Coordinates are normalized for different video resolutions
- **ID Preservation**: Object IDs remain consistent across resume sessions

## File Structure
```
tracking_storage/
├── last_dets.pkl      # Normalized detection data
└── last_frame.jpg     # Last processed frame
```

## Notes
- The `track_buffer` parameter affects how long tracks are maintained
- Storage files are overwritten each time new data is saved
- Only works for botsort currently

---
**Original README starts from here**
# BoxMOT: pluggable SOTA tracking modules for segmentation, object detection and pose estimation models

<div align="center">

  <img width="640"
       src="https://github.com/mikel-brostrom/boxmot/releases/download/v12.0.0/output_640.gif"
       alt="BoxMot demo">
  <br> <!-- one blank line -->

  [![CI](https://github.com/mikel-brostrom/yolov8_tracking/actions/workflows/ci.yml/badge.svg)](https://github.com/mikel-brostrom/yolov8_tracking/actions/workflows/ci.yml)
  [![PyPI version](https://badge.fury.io/py/boxmot.svg)](https://badge.fury.io/py/boxmot)
  [![downloads](https://static.pepy.tech/badge/boxmot)](https://pepy.tech/project/boxmot)
  [![license](https://img.shields.io/badge/license-AGPL%203.0-blue)](https://github.com/mikel-brostrom/boxmot/blob/master/LICENSE)
  [![python-version](https://img.shields.io/pypi/pyversions/boxmot)](https://badge.fury.io/py/boxmot)
  [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18nIqkBr68TkK8dHdarxTco6svHUJGggY?usp=sharing)
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8132989.svg)](https://doi.org/10.5281/zenodo.8132989)
  [![docker pulls](https://img.shields.io/docker/pulls/boxmot/boxmot?logo=docker)](https://hub.docker.com/r/boxmot/boxmot)
  [![discord](https://img.shields.io/discord/1377565354326495283?logo=discord&label=discord&labelColor=fff&color=5865f2)](https://discord.gg/3w4aYGbU)
</div>



## Introduction

This repository addresses the fragmented nature of the multi-object tracking (MOT) field by providing a standardized collection of pluggable, state-of-the-art trackers. Designed to seamlessly integrate with segmentation, object detection, and pose estimation models, the repository streamlines the adoption and comparison of MOT methods. For trackers employing appearance-based techniques, we offer a range of automatically downloadable state-of-the-art re-identification (ReID) models, from heavyweight ([CLIPReID](https://arxiv.org/pdf/2211.13977.pdf)) to lightweight options ([LightMBN](https://arxiv.org/pdf/2101.10774.pdf), [OSNet](https://arxiv.org/pdf/1905.00953.pdf)). Additionally, clear and practical examples demonstrate how to effectively integrate these trackers with various popular models, enabling versatility across diverse vision tasks.

<div align="center">

<!-- START TRACKER TABLE -->
| Tracker | Status  | HOTA↑ | MOTA↑ | IDF1↑ | FPS |
| :-----: | :-----: | :---: | :---: | :---: | :---: |
| [boosttrack](https://arxiv.org/abs/2408.13003) | ✅ | 69.253 | 75.914 | 83.206 | 25 |
| [botsort](https://arxiv.org/abs/2206.14651) | ✅ | 68.885 | 78.222 | 81.344 | 46 |
| [strongsort](https://arxiv.org/abs/2202.13514) | ✅ | 68.05 | 76.185 | 80.763 | 17 |
| [deepocsort](https://arxiv.org/abs/2302.11813) | ✅ | 67.796 | 75.868 | 80.514 | 12 |
| [bytetrack](https://arxiv.org/abs/2110.06864) | ✅ | 67.68 | 78.039 | 79.157 | 1265 |
| [ocsort](https://arxiv.org/abs/2203.14360) | ✅ | 66.441 | 74.548 | 77.899 | 1483 |

<!-- END TRACKER TABLE -->

<sub> NOTES: Evaluation was conducted on the second half of the MOT17 training set, as the validation set is not publicly available and the ablation detector was trained on the first half. We employed [pre-generated detections and embeddings](https://github.com/mikel-brostrom/boxmot/releases/download/v11.0.9/runs2.zip). Each tracker was configured using the default parameters from their official repositories. </sub>

</div>

</details>



## Why BOXMOT?

Multi-object tracking solutions today depend heavily on the computational capabilities of the underlying hardware. BoxMOT addresses this by offering a wide array of tracking methods tailored to accommodate diverse hardware constraints, ranging from CPU-only setups to high-end GPUs. Furthermore, we provide scripts designed for rapid experimentation, enabling users to save detections and embeddings once and subsequently reuse them with any tracking algorithm. This approach eliminates redundant computations, significantly speeding up the evaluation and comparison of multiple trackers.

## Installation

Install the `boxmot` package, including all requirements, in a Python>=3.9 environment:

```bash
pip install boxmot
```

BoxMOT provides a unified CLI `boxmot` with the following subcommands:

```bash
Usage: boxmot COMMAND [ARGS]...

Commands:
  track                  Run tracking only
  generate-dets-embs     Generate detections and embeddings
  generate-mot-results   Generate MOT evaluation results based on pregenerated detecions and embeddings
  eval                   Evaluate tracking performance using the official trackeval repository
  tune                   Tune tracker hyperparameters based on selected detections and embeddings
```

## YOLOv12 | YOLOv11 | YOLOv10 | YOLOv9 | YOLOv8 | RFDETR | YOLOX examples

<details>
<summary>Tracking</summary>

```bash
$ boxmot track --yolo-model rf-detr-base.pt     # bboxes only
  boxmot track --yolo-model yolox_s.pt          # bboxes only
  boxmot track --yolo-model yolo12n.pt         # bboxes only
  boxmot track --yolo-model yolo11n.pt         # bboxes only
  boxmot track --yolo-model yolov10n.pt         # bboxes only
  boxmot track --yolo-model yolov9c.pt          # bboxes only
  boxmot track --yolo-model yolov8n.pt          # bboxes only
                            yolov8n-seg.pt      # bboxes + segmentation masks
                            yolov8n-pose.pt     # bboxes + pose estimation
```

  </details>

<details>
<summary>Tracking methods</summary>

```bash
$ boxmot track --tracking-method deepocsort
                                 strongsort
                                 ocsort
                                 bytetrack
                                 botsort
                                 boosttrack
```

</details>

<details>
<summary>Tracking sources</summary>

Tracking can be run on most video formats

```bash
$ boxmot track --source 0                               # webcam
                        img.jpg                         # image
                        vid.mp4                         # video
                        path/                           # directory
                        path/*.jpg                      # glob
                        'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                        'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Select ReID model</summary>

Some tracking methods combine appearance description and motion in the process of tracking. For those which use appearance, you can choose a ReID model based on your needs from this [ReID model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO). These model can be further optimized for you needs by the [reid_export.py](https://github.com/mikel-brostrom/yolo_tracking/blob/master/boxmot/appearance/reid_export.py) script

```bash
$ boxmot track --source 0 --reid-model lmbn_n_cuhk03_d.pt               # lightweight
                                       osnet_x0_25_market1501.pt
                                       mobilenetv2_x1_4_msmt17.engine
                                       resnet50_msmt17.onnx
                                       osnet_x1_0_msmt17.pt
                                       clip_market1501.pt               # heavy
                                       clip_vehicleid.pt
                                      ...
```

</details>

<details>
<summary>Filter tracked classes</summary>

By default the tracker tracks all MS COCO classes.

If you want to track a subset of the classes that you model predicts, add their corresponding index after the classes flag,

```bash
boxmot track --source 0 --yolo-model yolov8s.pt --classes 16 17  # COCO yolov8 model. Track cats and dogs, only
```

[Here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) is a list of all the possible objects that a Yolov8 model trained on MS COCO can detect. Notice that the indexing for the classes in this repo starts at zero

</details>


</details>

<details>
<summary>Evaluation</summary>

Evaluate a combination of detector, tracking method and ReID model on standard MOT dataset or you custom one by

```bash
$ boxmot eval --yolo-model yolov8n.pt --reid-model osnet_x0_25_msmt17.pt --tracking-method deepocsort --verbose --source ./assets/MOT17-mini/train
$ boxmot eval --yolo-model yolov8n.pt --reid-model osnet_x0_25_msmt17.pt --tracking-method ocsort     --verbose --source ./tracking/val_utils/MOT17/train
```

add `--gsi` to your command for postprocessing the MOT results by gaussian smoothed interpolation. Detections and embeddings are stored for the selected YOLO and ReID model respectively. They can then be loaded into any tracking algorithm. Avoiding the overhead of repeatedly generating this data.
</details>


<details>
<summary>Evolution</summary>

We use a fast and elitist multiobjective genetic algorithm for tracker hyperparameter tuning. By default the objectives are: HOTA, MOTA, IDF1. Run it by

```bash
# saves dets and embs under ./runs/dets_n_embs separately for each selected yolo and reid model
$ boxmot generate-dets-embs --source ./assets/MOT17-mini/train --yolo-model yolov8n.pt yolov8s.pt --reid-model weights/osnet_x0_25_msmt17.pt
# evolve parameters for specified tracking method using the selected detections and embeddings generated in the previous step
$ boxmot tune --dets yolov8n --embs osnet_x0_25_msmt17 --n-trials 9 --tracking-method botsort --source ./assets/MOT17-mini/train
```

The set of hyperparameters leading to the best HOTA result are written to the tracker's config file.

</details>

<details>
<summary>Export</summary>

We support ReID model export to ONNX, OpenVINO, TorchScript and TensorRT

```bash
# export to ONNX
$ python3 boxmot/appearance/reid_export.py --include onnx --device cpu
# export to OpenVINO
$ python3 boxmot/appearance/reid_export.py --include openvino --device cpu
# export to TensorRT with dynamic input
$ python3 boxmot/appearance/reid_export.py --include engine --device 0 --dynamic
```

</details>


## Custom tracking examples

<div align="center">

| Example Description | Notebook |
|---------------------|----------|
| Torchvision bounding box tracking with BoxMOT | [![Notebook](https://img.shields.io/badge/Notebook-torchvision_det_boxmot.ipynb-blue)](examples/det/torchvision_boxmot.ipynb) |
| Torchvision pose tracking with BoxMOT | [![Notebook](https://img.shields.io/badge/Notebook-torchvision_pose_boxmot.ipynb-blue)](examples/pose/torchvision_boxmot.ipynb) |
| Torchvision segmentation tracking with BoxMOT | [![Notebook](https://img.shields.io/badge/Notebook-torchvision_seg_boxmot.ipynb-blue)](examples/seg/torchvision_boxmot.ipynb) |

</div>

## Contributors

<a href="https://github.com/mikel-brostrom/yolo_tracking/graphs/contributors ">
  <img src="https://contrib.rocks/image?repo=mikel-brostrom/yolo_tracking" />
</a>

## Contact

For BoxMOT bugs and feature requests please visit [GitHub Issues](https://github.com/mikel-brostrom/boxmot/issues).
For business inquiries or professional support requests please send an email to: box-mot@outlook.com

# YOLO Traffic Pipeline — 2-Stage Vehicle Detection, ALPR & Helmet Compliance

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](inference/CMakeLists.txt)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.19%2B-blueviolet.svg)](https://onnxruntime.ai/)
[![Status: Active Development](https://img.shields.io/badge/status-active%20development-yellow.svg)](#known-limitations)

### A C++ Pipeline for Vehicle Detection, License Plate Recognition, and Helmet Compliance

Traffic monitoring systems built as one monolithic model tend to fail in a specific way: they get good at the thing they saw most in training and quietly bad at everything else. A model trained mostly on close-up motorcycles struggles with wide-angle CCTV. A model trained to read plates struggles to first find them in a full traffic scene. This project avoids that by not asking one model to do everything.

Instead, detection happens in stages. A general-purpose model finds vehicles and people across the whole frame first. Only once a motorcycle or car is located does a second, specialized model get asked to find its plate — inside a small, already-cropped image, not the original frame. Text recognition and helmet classification follow the same pattern: narrow the problem down before asking a specialized model to solve it. The result runs as a single C++ binary against ONNX Runtime, with no Python interpreter required at inference time.

---

## Table of Contents

- [Philosophy](#philosophy)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Specification](#model-specification)
- [Project Layout](#project-layout)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Known Limitations](#known-limitations)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Philosophy

The pipeline is built around one recurring idea: **detect broad, then narrow, then specialize.** Stage 1 runs a single general-purpose detector across the full frame to answer a cheap question — what's here, and roughly where. Stage 2 only runs on the small region that answer points to, and only for the objects that need it. A car's plate gets looked for inside the car's own bounding box, not the whole 1920×1080 frame; a rider's helmet only gets classified if a person's box actually overlaps a motorcycle's box. This keeps the expensive, fine-grained models fast because they never have to search — they're handed a crop where the answer is already close by.

Model 1 currently runs as the stock COCO-pretrained YOLO11n, not a custom-trained model. That was a deliberate choice, not a placeholder: COCO's `person`, `car`, `motorcycle`, `bus`, and `truck` classes are already trained on a far larger and more varied set of poses and angles than a small custom dataset could realistically match early on, and it sidesteps a real risk this project ran into during dataset planning — a custom vehicle model trained only on front-facing motorcycles and rider-posed people would have quietly failed on anything else (a motorcycle seen from behind, a pedestrian on a sidewalk). Swapping Model 1 back to a custom-trained model later doesn't require touching any code — see [Configuration](#configuration).

Object identity across frames is handled by ByteTrack rather than by re-running the expensive stages on every frame. Once a vehicle has a track ID, its plate text and helmet status are computed once and cached against that ID — a motorcycle visible for 100 frames triggers one OCR call, not a hundred.

None of the threshold values a real deployment needs to tune — confidence cutoffs, NMS, tracker sensitivity — are hardcoded. They live in an external config file read at startup, so calibration is an edit-and-rerun loop, not an edit-and-recompile one.

---

## Architecture

```mermaid
flowchart TD
    A[Frame: image, video, or RTSP stream] --> B[Stage 1: Model 1 full-frame detection]
    B -->|boxes + class labels| C{Classify each detection}

    C -->|person| D[Kept for driver-overlap check]
    C -->|car / bus / truck| E[Crop vehicle region]
    C -->|motorcycle| F[Crop motorcycle region]

    F --> G[ByteTrack: assign persistent track ID]
    G --> H[Stage 2a: plate detector on crop]
    E --> H

    H -->|plate box found| I[Crop top 70% of plate region]
    I --> J[Stage 2b: CTC text recognizer]

    F --> K{Does a person box overlap this motorcycle?}
    K -->|yes| L[Stage 2c: Model 2 helmet classifier]
    K -->|no| M[Status: no_rider]

    J --> N[Draw boxes, labels, and text on frame]
    L --> N
    M --> N
    D --> N
```

**Stage 1 — full-frame detection.** One forward pass identifies every vehicle and person in the scene. Cars, buses, and trucks are drawn and tracked as vehicles but don't go through plate detection unless they carry a plate class match; motorcycles proceed to Stage 2 in full, since they're the only class that also needs helmet compliance checking.

**Stage 2a/2b — plate detection and reading.** A vehicle's own crop is handed to a small plate detector, and the resulting plate box is handed to a CTC-based text recognizer. Indonesian plates commonly carry two lines — the plate number and a smaller expiration date below it — which confuses a single-line CTC model if fed directly, so the crop is trimmed to its top 70% before recognition.

**Stage 2c — helmet compliance.** Only triggered when a `person` detection's box overlaps a `motorcycle` detection's box from Stage 1 — there's no dedicated "rider" class, so overlap is the proxy for "this motorcycle currently has someone on it."

**Tracking and caching.** ByteTrack assigns each motorcycle a track ID that persists as long as it stays in frame. Plate text and helmet status are computed once per track ID and reused on every subsequent frame that ID appears in, rather than re-run per frame.

---

## Installation

### Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| CMake | 3.16+ | Build system generator |
| GCC/G++ | 9.0+ (C++17 support) | Compiler |
| OpenCV | 4.0+ | Image I/O, preprocessing, visualization |
| Eigen3 | 3.3+ | Linear algebra (ByteTrack dependency) |
| Git | — | Dependency fetch via CMake FetchContent |

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential libopencv-dev libeigen3-dev git
```

### ONNX Runtime

The C++ build links against a prebuilt ONNX Runtime distribution, which isn't vendored in this repository (see [.gitignore](.gitignore) — model weights and third-party binaries are deliberately kept out of version control). Download and extract it once:

```bash
mkdir -p third_party && cd third_party
curl -L -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz
tar xzf ort.tgz && rm ort.tgz
cd ..
```

### Building

```bash
cd inference
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

ByteTrack-cpp is fetched automatically by CMake on first configure (via `FetchContent`), so it doesn't need a separate install step. The build produces a single binary at `inference/build/yolo`.

### Models

Model weights are not distributed in this repository — they're trained and exported separately (see [training/](training/)) and placed manually into `models/`:

```
models/
├── model1.onnx           # Stage 1 detector (COCO pretrained, or a custom-trained equivalent)
├── model1_classes.txt    # Class names for model1.onnx, one per line, in output order
├── plate.onnx            # Stage 2a plate detector
├── plate_rec.onnx        # Stage 2b CTC text recognizer
├── char_dict.txt         # Character dictionary for plate_rec.onnx's CTC decoding
└── model2.onnx            # Stage 2c helmet classifier
```

Two ready-to-copy templates for `model1_classes.txt` are in [inference/config/](inference/config/): one for the stock COCO model, one for a custom 3-class (`car`, `motorcycle`, `person`) model. See that folder's [README](inference/config/README.md) for the naming rules the file needs to follow.

### Python (optional, training only)

The C++ binary has no Python dependency at runtime. A virtual environment is only needed for the training and export scripts under `training/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

> Run the binary from inside `inference/` — model and data paths are resolved relative to that directory (`../models/`, `../data/`).

### Single image

```bash
cd inference
./build/yolo ../data/input/images/2.jpg
```

```text
Loading Model 1 (8 classes dari model1_classes.txt)...
Loading Plate model (2 classes: plate, vehicle)...
Loading Model 2 (helmet, no_helmet)...
Loading Plate text recognizer (CTC)...
  [Vehicle Car ID:0] Plate:B1483FYJ
  [Vehicle Bus ID:1] Plate:-
Saved ../data/output/2.jpg | cars=1 buses=1 trucks=0 motorcycles=0 persons=1
```

The annotated result is written to `../data/output/` under the same filename.

### Video, webcam, or RTSP stream

```bash
./build/yolo path/to/traffic_video.mp4
./build/yolo 0                                              # local webcam, device index 0
./build/yolo "rtsp://admin:password@192.168.1.64:554/stream"
```

Press `q` or `Esc` to exit. Track IDs, per-vehicle plate text, and helmet status render live on the frame; an FPS counter sits in the top-left corner.

---

## Configuration

Every threshold a real deployment needs to calibrate lives in an optional file, `models/pipeline_config.txt`, read once at startup — nothing here requires touching or recompiling `main.cpp`:

```bash
cp inference/config/pipeline_config.example.txt models/pipeline_config.txt
```

| Key | Controls |
|---|---|
| `model1_conf_thresh`, `model1_nms_thresh` | Confidence and NMS thresholds for Stage 1 detection |
| `plate_conf_thresh`, `plate_nms_thresh` | Same, for plate detection |
| `model2_conf_thresh`, `model2_nms_thresh` | Same, for helmet classification |
| `bytetrack_track_buffer` | Frames a lost track is kept alive before being dropped |
| `bytetrack_track_thresh`, `bytetrack_high_thresh`, `bytetrack_match_thresh` | ByteTrack's internal matching thresholds |
| `plate_padding_ratio` | Padding added around a detected plate box before OCR crop |
| `min_driver_overlap_iou` | Minimum IoU between a person box and a motorcycle box to count as "has a rider" |

If the file doesn't exist, every value falls back to a built-in default — the pipeline runs identically to before this file existed. Malformed lines or non-numeric values are logged as a warning and skipped rather than treated as a fatal error; everything else in the file still takes effect.

Class mapping for Model 1 works the same way: `models/model1_classes.txt` lists class names in the order the model outputs them, and `main.cpp` resolves `person`/`motorcycle`/`car`/`bus`/`truck` by name at startup rather than by hardcoded index. Switching Model 1 between the COCO pretrained model and a custom-trained one is a matter of swapping this file, not editing code.

---

## Model Specification

| Model | Architecture | Input | Output | Role |
|---|---|---|---|---|
| `model1.onnx` | YOLO11n | 640×640 | Configurable via `model1_classes.txt` (80 for COCO) | Full-frame vehicle & person detection |
| `plate.onnx` | YOLO11n | 640×640 | 2 classes (`plate`, `vehicle`) | Plate localization within a vehicle crop |
| `plate_rec.onnx` | CRNN + CTC | 48×W (dynamic width) | Vocabulary sized to `char_dict.txt` | Character recognition on a plate crop |
| `model2.onnx` | YOLO11n | 640×640 | 2 classes (`helmet`, `no_helmet`) | Helmet presence on a motorcycle crop |

Accuracy and inference-time numbers aren't included here on purpose — this pipeline hasn't gone through a proper accuracy evaluation pass yet (see [Known Limitations](#known-limitations)). Numbers estimated from a handful of manual tests would look precise without actually meaning anything; real figures will go here once there's a real benchmark behind them.

---

## Project Layout

```
yolo-project/
├── data/
│   ├── input/images/       Sample test images
│   └── output/              Annotated inference results
├── inference/
│   ├── CMakeLists.txt        CMake build config, ByteTrack-cpp via FetchContent
│   ├── config/                 Class-list and pipeline-config templates + docs
│   └── src/
│       ├── detector.h/.cpp      Generic YOLO ONNX Runtime wrapper
│       ├── recognizer.h/.cpp    CTC preprocessing and greedy decode
│       ├── config.h/.cpp        pipeline_config.txt parser
│       └── main.cpp              Pipeline orchestration
├── models/                    Model weights and class/dict/config files (not versioned)
├── training/                  Training, export, and debugging scripts (Python)
├── requirements.txt          Python dependencies for training/export only
└── CHANGELOG.md
```

---

## Development

```bash
# Rebuild after a source change
cd inference/build && make -j$(nproc)

# Force a clean CMake reconfigure (e.g. after changing CMakeLists.txt)
rm -rf inference/build && cd inference && mkdir build && cd build && cmake .. && make -j$(nproc)
```

There's no automated test suite yet. Manual verification currently means: run the binary against a known image or video, and check the annotated output and console log by eye. Adding real tests around `detector.cpp`'s decode/NMS logic and `recognizer.cpp`'s CTC decode is on the list — see [Known Limitations](#known-limitations).

---

## Troubleshooting

**`onnxruntime_c_api.h: No such file or directory`** — ONNX Runtime isn't where `CMakeLists.txt` expects it. Confirm `third_party/onnxruntime-linux-x64-*/include/` exists (see [Installation](#installation)); the version in the folder name has to match what `CMakeLists.txt` references.

**`undefined reference to cv::imread`** — OpenCV isn't linked. Confirm `libopencv-dev` is installed and `pkg-config --modversion opencv4` resolves.

**`Gagal load model ONNX: ... File doesn't exist`** — the binary is being run from the wrong working directory, or a model file is missing from `models/`. Model paths are resolved relative to `inference/build/`, so run the binary from `inference/` (see [Usage](#usage)), and confirm every file listed under [Installation → Models](#installation) is actually present.

**Class mismatch — a car gets labeled as a motorcycle, or vice versa** — `models/model1_classes.txt` doesn't match the actual class order `model1.onnx` outputs. This exact bug happened during development: Model 1 was switched to COCO pretrained, but the class-id constants in the code were still written for a 3-class custom scheme, so index 0 (COCO `person`) got treated as `car`. Fixed by resolving class ids by name at runtime instead of by hardcoded index — see [Configuration](#configuration) and the `[0.2.0]` entry in [CHANGELOG.md](CHANGELOG.md).

**Crash with no clear error message** — build with debug symbols and get a backtrace:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j$(nproc)
gdb ./yolo
(gdb) run ../data/input/images/1.jpg
(gdb) bt
```

---

## Known Limitations

- **No accuracy evaluation yet.** The pipeline runs end-to-end without crashing on real models, but detection/OCR/helmet-classification accuracy hasn't been systematically measured against ground truth. Threshold values in [`pipeline_config.example.txt`](inference/config/pipeline_config.example.txt) are reasonable starting points, not calibrated numbers.
- **GUI/video mode is comparatively under-tested.** Development and validation so far has mostly exercised single-image mode; `cv::imshow`-based video/camera mode works but has seen less real-world testing.
- **`plate.onnx` covers cars in addition to motorcycles**, using a 2-class dataset (`plate`, `vehicle`) rather than the earlier motorcycle-only model this project also has available. This trades some motorcycle-specific accuracy for broader vehicle coverage — worth revisiting once real accuracy numbers exist.
- **No speed estimation.** Camera calibration and pixel-to-real-world distance conversion — part of the original project goal — hasn't been started.
- **No automated tests.** Correctness is currently verified by manual inspection of annotated output, not by a test suite.

---

## Acknowledgments

Built on [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) for detection, [ONNX Runtime](https://onnxruntime.ai/) for cross-platform inference, [ByteTrack](https://github.com/Vertical-Beach/ByteTrack-cpp) for multi-object tracking, and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)'s recognition model (exported to ONNX) for plate text recognition.

---

## License

MIT — see [LICENSE](LICENSE) for the full text.

---

*A traffic monitoring pipeline built to detect broad, then narrow, then specialize.*

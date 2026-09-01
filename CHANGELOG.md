# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- External config file (`models/pipeline_config.txt`) for every calibration threshold: confidence and NMS thresholds for all three detection models, ByteTrack's `track_buffer`/`track_thresh`/`high_thresh`/`match_thresh`, plate crop padding ratio, and the minimum IoU required between a person box and a motorcycle box to count as "has a rider". The file is optional — if absent, every value falls back to the same defaults the pipeline used before this existed. Malformed lines or non-numeric values are logged as a warning (with line number) and skipped rather than treated as fatal; the parser is a small hand-written key=value reader, not a new JSON/YAML dependency.

## [0.2.0] - 2026-08-21

### Changed
- Class id resolution for Model 1 (`person`, `motorcycle`, `car`, `bus`, `truck`) moved from hardcoded integer constants in `main.cpp` to `YoloDetector::classIndexByName()`, which looks up a class's index by name at runtime against the list loaded from `models/model1_classes.txt`. Switching `model1.onnx` between the COCO-pretrained model and a custom-trained one is now a matter of swapping that text file — no code change or recompile required. Two templates ship in `inference/config/` (`coco80_classes.txt`, `custom3_classes.txt`).
- `bus` and `truck` are now optional in Model 1's class list. If the loaded model doesn't have them (e.g. a 3-class custom model), `classIndexByName()` fails for just those two and the ids are set to `-1`, which never matches a real detection — the rest of the pipeline runs unaffected.

### Fixed
- **Critical**: Model 1 had been switched from a custom 3-class scheme (`car=0, motorcycle=1, person=2`) to the stock COCO-pretrained YOLO11n, but the class-id constants in `main.cpp` were never updated to match COCO's actual ordering (`person=0, bicycle=1, car=2, motorcycle=3, ...`). Every `car` detection (COCO id 2) was read against the old constant `MOTORCYCLE=1`, so cars were silently processed as motorcycles — including being sent through helmet-compliance logic they had no business going through. Root-caused by comparing detection output against `data1.yaml`'s actual class order rather than assumed order; fixed by removing hardcoded ids entirely (see Changed, above) rather than just correcting the numbers, so the same class of bug can't recur on the next model swap.

## [0.1.0] - 2026-08-20

### Added
- Initial 2-stage detection pipeline: Model 1 (full-frame vehicle and person detection) feeds ByteTrack for persistent per-vehicle tracking; motorcycles and other vehicles are cropped and passed to a dedicated plate detector, and motorcycles with an overlapping person box are additionally passed to a helmet classifier.
- License plate OCR implemented natively in C++: PaddleOCR's `PP-OCRv6_medium_rec` recognition model exported to ONNX (`paddle2onnx`) and paired with a hand-written CTC greedy decoder — no PaddlePaddle or Python dependency at inference time.
- Top-70% plate crop step before OCR, to isolate the main plate number from the smaller expiration-date line printed below it on Indonesian plates, which otherwise confused the single-line CTC decoder.
- ByteTrack-cpp integrated via CMake `FetchContent`, avoiding a manual vendoring step.
- `YoloDetector`: a single generic ONNX Runtime wrapper (letterbox preprocessing, decode, NMS) reused across all three YOLO-based models instead of duplicated per-model code.
- Per-track-id caching for both plate text and helmet status, so a vehicle visible across many frames triggers one OCR/classification pass rather than one per frame.

### Fixed
- ONNX Runtime C++ API dangling-pointer bug: `session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo()` returns a view into the temporary `TypeInfo` object, which is destroyed at the end of the statement — calling `.GetShape()` on the result in a later statement read freed memory and crashed with `std::length_error` inside `std::vector`'s allocator, not inside anything that looked related to ONNX Runtime. Root-caused with `gdb catch throw`. Fixed by keeping the parent `TypeInfo` alive in a named variable for as long as its derived shape info is in use.

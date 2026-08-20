import cv2
import time
from ultralytics import YOLO
from paddleocr import PaddleOCR
import os

# --- Mapping class id (HARUS sinkron dengan data1.yaml / data2.yaml di notebook training) ---
# Model 1: car, motorcycle, person
MODEL1_CAR_ID = 0
MODEL1_MOTORCYCLE_ID = 1
MODEL1_PERSON_ID = 2

# Model 2: helmet, no_helmet
MODEL2_HELMET_ID = 0
MODEL2_NO_HELMET_ID = 1


def boxes_overlap(box_a, box_b):
    """Cek apakah 2 box (x1,y1,x2,y2) saling overlap.
    Simplifikasi dari logika 'driver = person overlap dengan vehicle box' di project brief.
    CATATAN: ini cek overlap sederhana, bukan IoU threshold -> kalau ada pejalan kaki
    yang kebetulan lewat pas overlap sama box motor, bisa salah dianggap driver.
    Cukup buat MVP, perlu diperketat (mis. pakai IoU minimum) kalau false-positive banyak."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    return inter_x2 > inter_x1 and inter_y2 > inter_y1


def process_dashcam_video(input_path, output_path):
    print("Loading models for Dashcam Pipeline (2-stage architecture)...")
    # Model 1 jalan full-frame: car, motorcycle, person
    vehicle_model = YOLO("../models/model1.onnx")
    # Plate: TIDAK ditraining ulang, reuse model yang sudah divalidasi (mAP 97.9%)
    plate_model = YOLO("../models/plate.pt")
    # Model 2 jalan di crop motor+rider: helmet, no_helmet
    helmet_model = YOLO("../models/model2.onnx")
    ocr = PaddleOCR(use_textline_orientation=False, lang='en', enable_mkldnn=False)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing dashcam video: {width}x{height} @ {fps}fps ({total_frames} frames)")

    # --- MEMORI (Caching), key = track_id dari MOTORCYCLE (bukan dari plate lagi) ---
    ocr_memory = {}      # track_id -> teks plat
    helmet_memory = {}   # track_id -> "helmet" / "no_helmet"

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # --- STAGE 1: Model 1 full-frame, dengan tracking ---
        results = vehicle_model.track(frame, conf=0.25, persist=True, tracker="bytetrack.yaml", verbose=False)

        person_boxes = []
        motorcycle_detections = []  # list of (track_id, (x1,y1,x2,y2))

        for r in results:
            if r.boxes.id is None:
                continue
            for box, track_id_tensor in zip(r.boxes, r.boxes.id):
                cls = int(box.cls[0])
                track_id = int(track_id_tensor.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls == MODEL1_PERSON_ID:
                    person_boxes.append((x1, y1, x2, y2))
                elif cls == MODEL1_MOTORCYCLE_ID:
                    motorcycle_detections.append((track_id, (x1, y1, x2, y2)))
                elif cls == MODEL1_CAR_ID:
                    # car berhenti di Model 1, tidak masuk plate/helmet (sesuai keputusan arsitektur)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # --- STAGE 2: per motorcycle -> crop-then-detect plate, dan (kalau ada rider) helmet ---
        for track_id, (mx1, my1, mx2, my2) in motorcycle_detections:
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 255, 0), 2)

            crop_x1, crop_y1 = max(0, mx1), max(0, my1)
            crop_x2, crop_y2 = min(width, mx2), min(height, my2)
            motor_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if motor_crop.size == 0:
                continue

            # --- Plate: crop-then-detect (bukan full-frame lagi) ---
            plate_text = ocr_memory.get(track_id, None)
            if plate_text is None:
                plate_results = plate_model.predict(motor_crop, conf=0.25, verbose=False)
                for pr in plate_results:
                    if len(pr.boxes) == 0:
                        continue
                    # ambil box plate dengan confidence tertinggi di crop ini
                    best_plate_box = max(pr.boxes, key=lambda b: float(b.conf[0]))
                    px1, py1, px2, py2 = map(int, best_plate_box.xyxy[0])
                    plate_w = px2 - px1

                    # threshold disesuaikan krn sudah di-crop motor (skala beda dari full-frame)
                    if plate_w > 40:
                        plate_crop = motor_crop[py1:py2, px1:px2]
                        ch, cw = plate_crop.shape[:2]
                        if cw > 0 and ch > 0:
                            if cw < 200:
                                scale = 200 / cw
                                plate_crop = cv2.resize(plate_crop, None, fx=scale, fy=scale,
                                                         interpolation=cv2.INTER_CUBIC)
                            ocr_result = ocr.predict(plate_crop)
                            new_text = ""
                            if ocr_result and ocr_result[0]:
                                for res in ocr_result[0]:
                                    text = res[1][0]
                                    ocr_conf = res[1][1]
                                    if ocr_conf > 0.5:
                                        new_text += text + " "
                            new_text = new_text.strip()
                            if new_text:
                                ocr_memory[track_id] = new_text
                                plate_text = new_text

            # --- Helmet: cuma dicek kalau ada person yang overlap (driver) ---
            has_driver = any(boxes_overlap((mx1, my1, mx2, my2), pbox) for pbox in person_boxes)
            helmet_status = helmet_memory.get(track_id, None)

            if has_driver and helmet_status is None:
                helmet_results = helmet_model.predict(motor_crop, conf=0.25, verbose=False)
                for hr in helmet_results:
                    if len(hr.boxes) == 0:
                        continue
                    best_helmet_box = max(hr.boxes, key=lambda b: float(b.conf[0]))
                    hcls = int(best_helmet_box.cls[0])
                    helmet_status = "helmet" if hcls == MODEL2_HELMET_ID else "no_helmet"
                    helmet_memory[track_id] = helmet_status

            # --- Visualisasi ---
            label_parts = [f"ID:{track_id}"]
            label_parts.append(plate_text if plate_text else "[Plat: mencari...]")
            if has_driver:
                label_parts.append(f"[{helmet_status if helmet_status else 'cek helm...'}]")
            else:
                label_parts.append("[Tanpa rider]")
            label = " ".join(label_parts)

            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (mx1, my1 - text_h - 10), (mx1 + text_w, my1), (0, 255, 0), -1)
            cv2.putText(frame, label, (mx1, my1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out.write(frame)

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {frame_count}/{total_frames} frames ({(frame_count/elapsed):.1f} fps) | "
                  f"Unique Plates OCR-ed: {len(ocr_memory)} | Unique Helmet checks: {len(helmet_memory)}")

    cap.release()
    out.release()
    print(f"\nDone! Output saved to: {output_path}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Total unique plates recognized: {len(ocr_memory)}")
    print(f"Total unique helmet checks: {len(helmet_memory)}")


if __name__ == "__main__":
    input_video = "../data/input/videos/test_dashcam.mp4"
    output_video = "../data/output/videos/output_dashcam.mp4"

    if not os.path.exists(input_video):
        print(f"Video tidak ditemukan: {input_video}")
        print("Silakan upload video rekaman dashcam ke folder data/input/videos/ dengan nama test_dashcam.mp4")
    else:
        process_dashcam_video(input_video, output_video)

# video_pipeline.py
import cv2
import time
from ultralytics import YOLO
from paddleocr import PaddleOCR

def process_video(input_path, output_path):
    # --- Load Models ---
    print("Loading models...")
    plate_model = YOLO("../models/license_plate.pt")
    # Disable textline orientation as plates are usually horizontal
    ocr = PaddleOCR(use_textline_orientation=False, lang='en', enable_mkldnn=False)

    # --- Video Setup ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Get video properties for output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {width}x{height} @ {fps}fps ({total_frames} frames)")
    
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # --- Deteksi Plat ---
        results = plate_model.predict(frame, conf=0.25, verbose=False)
        
        for r in results:
            for box in r.boxes:
                # Fix 1: Filter hanya class plate (class=0)
                cls = int(box.cls[0])
                if cls != 0:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Gambar kotak untuk plat
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Fix 2: Tambah padding
                pad = 10
                crop_x1 = max(0, x1 - pad)
                crop_y1 = max(0, y1 - pad)
                crop_x2 = min(width, x2 + pad)
                crop_y2 = min(height, y2 + pad)

                # Crop otomatis
                plate_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                # Fix 3: Resize crop jika kekecilan
                crop_h, crop_w = plate_crop.shape[:2]
                if crop_w > 0 and crop_h > 0:
                    if crop_w < 200:
                        scale = 200 / crop_w
                        plate_crop = cv2.resize(plate_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

                    # --- OCR ---
                    ocr_result = ocr.predict(plate_crop)
                    
                    plate_text = ""
                    if ocr_result and ocr_result[0]:
                        for res in ocr_result[0]:
                            text = res[1][0]
                            ocr_conf = res[1][1]
                            # Jika confidence OCR lumayan bagus
                            if ocr_conf > 0.5:
                                plate_text += text + " "
                    
                    plate_text = plate_text.strip()
                    
                    # Tulis teks hasil OCR di atas kotak plat
                    if plate_text:
                        label = f"{plate_text} ({conf:.2f})"
                        # Background teks supaya mudah dibaca
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Tulis frame ke file output
        out.write(frame)
        
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {frame_count}/{total_frames} frames ({(frame_count/elapsed):.1f} fps)")

    # Cleanup
    cap.release()
    out.release()
    print(f"\nDone! Output saved to: {output_path}")
    print(f"Total time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    # Path video input & output
    input_video = "../data/input/videos/test_video.mp4"    # <--- GANTI DENGAN NAMA VIDEO ANDA
    output_video = "../data/output/videos/output_video.mp4" 
    
    import os
    if not os.path.exists(input_video):
        print(f"Video tidak ditemukan: {input_video}")
        print("Silakan upload/simpan video di folder data/input/videos/ dengan nama test_video.mp4")
    else:
        process_video(input_video, output_video)

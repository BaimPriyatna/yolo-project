from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2

# --- Load kedua model ---
plate_model = YOLO("../models/license_plate.pt")
ocr = PaddleOCR(use_textline_orientation=False, lang='en', enable_mkldnn=False)

# --- Deteksi plat otomatis ---
image_path = "../data/input/images/2.jpg"
results = plate_model.predict(image_path, conf=0.25)

img = cv2.imread(image_path)
h, w = img.shape[:2]

for r in results:
    for box in r.boxes:
        # Fix 1: Filter hanya class plate (class=0), skip vehicle dll
        cls = int(box.cls[0])
        if cls != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        print(f"Plat terdeteksi (confidence={conf:.2f}) di [{x1},{y1},{x2},{y2}]")

        # Fix 2: Tambah padding agar crop plat lebih besar untuk OCR
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        # Crop otomatis dari hasil deteksi
        plate_crop = img[y1:y2, x1:x2]

        # Fix 3: Resize crop agar minimal 200px lebar untuk akurasi OCR
        crop_h, crop_w = plate_crop.shape[:2]
        if crop_w < 200:
            scale = 200 / crop_w
            plate_crop = cv2.resize(plate_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        cv2.imwrite("../data/output/crops/plate_crop.jpg", plate_crop)

        # OCR baca teks di crop-nya
        ocr_result = ocr.predict("../data/output/crops/plate_crop.jpg")
        for res in ocr_result:
            print("Teks terbaca:", res['rec_texts'])
            print("Confidence OCR:", res['rec_scores'])
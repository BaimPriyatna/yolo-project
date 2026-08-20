from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2

plate_model = YOLO('../models/license_plate.pt')
ocr = PaddleOCR(use_textline_orientation=False, lang='en', enable_mkldnn=False)

image_path = '../data/input/images/1.jpg'
print(f'Testing {image_path}')

results = plate_model.predict(image_path, conf=0.10) # Lowering conf to see if it detects anything
img = cv2.imread(image_path)
print('Image shape:', img.shape)

detected = False
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f'Found box: class={cls}, conf={conf:.2f}')
        if cls == 0:
            detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f'Plate coordinates: [{x1},{y1},{x2},{y2}]')
            
            # Simple crop
            plate_crop = img[y1:y2, x1:x2]
            
            ocr_result = ocr.predict(plate_crop)
            print('OCR Result:', ocr_result)

if not detected:
    print('No plate (class 0) detected by YOLO even with conf=0.10.')

from ultralytics import YOLO
import cv2

plate_model = YOLO("../models/license_plate_motorcycle.pt")
image_path = "../data/input/images/1.jpg"
results = plate_model.predict(image_path, conf=0.25)

img = cv2.imread(image_path)

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = r.names[class_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imwrite("../data/output/debug/debug_detection_motorcycle.jpg", img)
print("Saved debug_detection_motorcycle.jpg")
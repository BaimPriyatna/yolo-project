from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # base pretrained, transfer learning
results = model.train(data="Indonesia-License-Plate-2/data.yaml", epochs=50, imgsz=640, device="cpu")
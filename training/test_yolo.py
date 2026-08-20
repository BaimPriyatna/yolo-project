from ultralytics import YOLO

model = YOLO("yolo11n.yaml")
results = model("https://ultralytics.com/images/bus.jpg")

print("success")
from ultralytics import YOLO

model = YOLO("yolov8s-p2-bigate.yaml")

model.load("yolov8s.pt")

model.info()

print("\nYOLOv8s-P2-BiGate build success.")
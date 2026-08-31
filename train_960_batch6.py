
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    data="VisDrone.yaml",
    epochs=150,
    imgsz=960,
    batch=6,
    device=0,
    workers=8,
    optimizer="auto",
    amp=True,
    patience=150,
    close_mosaic=10,
    project="runs/detect",
    name="v8s_960_batch6_150e",
)

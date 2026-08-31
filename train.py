from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(
    data="VisDrone.yaml",
    epochs=150,
    imgsz=1280,
    batch=8,
    device=0,
    workers=8,
    optimizer="auto",
    amp=True,
    cache=False,
    patience=30,
    close_mosaic=10,
    project="runs/detect",
    name="v8s_1280_batch8_150e",
)

from ultralytics import YOLO


# ============================================================
# 1. Build YOLOv8s-P2-BiGate
# ============================================================

model = YOLO("yolov8s-p2-bigate.yaml")


# ============================================================
# 2. Load the same COCO pretrained weights
# ============================================================

model.load("yolov8s.pt")


# ============================================================
# 3. Train
# ============================================================

results = model.train(
    data="VisDrone.yaml",

    imgsz=960,
    epochs=150,
    batch=6,

    device=0,
    workers=8,

    optimizer="auto",

    amp=True,
    pretrained=True,

    cache=False,

    close_mosaic=10,

    patience=150,

    seed=0,
    deterministic=True,

    project="/root/rivermind-data/yolo/ultralytics-main/runs/detect",

    name="v8s_p2_bigate_960_batch6_150e",
)
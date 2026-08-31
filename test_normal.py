from ultralytics import YOLO

model = YOLO(
    "/root/rivermind-data/yolo/ultralytics-main/"
    "runs/detect/v8s_960_batch6_150e/weights/best.pt"
)

metrics = model.val(
    data="VisDrone.yaml",
    imgsz=960,
    batch=6,
    device=0,
    plots=True,
    project="/root/rivermind-data/yolo/ultralytics-main/runs/detect",
    name="normal",
)
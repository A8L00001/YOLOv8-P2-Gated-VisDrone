from ultralytics import YOLO

model = YOLO(
    "runs/detect/v8s_p2_gate_960_batch6_150e/weights/best.pt"
)

model.val(
    data="VisDrone.yaml",
    imgsz=960,
    batch=6,
    device=0,
    plots=True,
    project="/root/rivermind-data/yolo/ultralytics-main/runs/final_val",
    name="p2_gate",
)
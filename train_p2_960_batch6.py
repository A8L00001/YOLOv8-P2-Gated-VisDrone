from ultralytics import YOLO


# 1. 使用官方 P2 配置构建 YOLOv8s-P2
model = YOLO("yolov8s-p2.yaml")

# 2. 加载标准 YOLOv8s COCO 预训练权重
model.load("yolov8s.pt")

# 3. 在 VisDrone 上训练
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
    patience=150,
    close_mosaic=10,

    project="/root/rivermind-data/yolo/ultralytics-main/runs/detect",
    name="v8s_p2_960_batch6_150e",
)
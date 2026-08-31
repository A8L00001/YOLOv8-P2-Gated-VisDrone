from ultralytics import YOLO

model = YOLO("runs/detect/train-2/weights/best.pt")

metrics = model.val(
    data="VisDrone.yaml",
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    max_det=300,
    workers=0,
    plots=True,
    name="visdrone_test",
)

print(metrics.results_dict)
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

results = model.predict(source=0, show=True)  # predict on an image
import ultralytics
from ultralytics import YOLO
model = YOLO("yolo11m.pt")
results = model.train(
    data="Malaysia-road-sign-dataset-1/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    amp=True,
    patience=15,
    plots=True,
    save=True
)
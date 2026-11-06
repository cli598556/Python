# using a dynamic video to track
from ultralytics import YOLO
model = YOLO("runs/detect/train2/weights/best.pt")
results = model.track(source="c0791d83f7e86c88cc84e4b49b658698.mp4",
                      show=True,
                      tracker="bytetrack.yaml",
                      save=True,
                      project="My_PPT_Demo",
                      name="Tracking_Result"
)


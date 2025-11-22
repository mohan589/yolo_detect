from ultralytics import YOLO

model = YOLO("runs/detect/train10/weights/best.pt")

model.predict(
  source="test/",
  save=True,
  conf=0.25
)

print("Inference Completed!")
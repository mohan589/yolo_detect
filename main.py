from ultralytics import YOLO

# Load a pre-trained YOLOv12-L model
model = YOLO('yolov12l.pt')

# Perform inference on an image
results = model('path/to/your/image.jpg')
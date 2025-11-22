from ultralytics import YOLO

def main():
  # Choose one of the YOLOv12 models:
  model = YOLO("yolo12l.pt")  # n, s, m, l, x

  model.train(
    data="datasets.yaml",
    imgsz=640,
    epochs=100,
    batch=16,
    patience=50,
    device='cpu',       # GPU 0
    workers=8,      # dataloader workers
    lr0=0.01,       # initial learning rate
    pretrained=True
  )

  print("Training Completed!")
  print("Best weights saved at: runs/detect/train/weights/best.pt")

if __name__ == "__main__":
  main()

Training command:

yolo detect train model=yolo12m.pt data=my_dataset.yaml epochs=100 imgsz=640

Example:
yolo detect train \
  model=yolo12l.pt \
  data=my_dataset.yaml \
  epochs=150 \
  imgsz=640 \
  batch=16 \
  device=0


Running Commands

uv run train.py

uv run inference.py
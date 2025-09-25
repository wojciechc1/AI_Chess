from ultralytics import YOLO

model = YOLO('runs/detect/train3/weights/last.pt')

model.train(
    data='./dataset/data.yaml',
    imgsz=640,
    epochs=30
)
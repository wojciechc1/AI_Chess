from ultralytics import YOLO

model = YOLO("board_seg/exp12/weights/best.pt")

model.train(
    data="Dataset1/data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    project="board_seg",
    name="exp1",
)
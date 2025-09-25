from ultralytics import YOLO
import cv2
import numpy as np


def detect_and_visualize(img_resized, model_path="best.pt", target_size=640, conf=0.3):

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    model = YOLO(model_path)
    results = model(img_rgb, imgsz=target_size, conf=conf)

    r = results[0]  # pierwsza (i jedyna) klatka

    return r, model.names[int(cls)]



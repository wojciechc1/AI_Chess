from board_segmentation import predict_board_layout, get_grid
import cv2
from ultralytics import YOLO
import time
import numpy as np

# config
TARGET_SIZE = 640
CONFIDENCE_THRESHOLD = 0.3
dst_pts = np.array([[0, 0], [TARGET_SIZE, 0], [TARGET_SIZE, TARGET_SIZE], [0, TARGET_SIZE]], dtype="float32")


# image loading
img = cv2.imread("img_4.jpg")
img_resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)


# yolo models
model = YOLO("board_seg/exp13/weights/best.pt")
model1 = YOLO("runs/detect/train4/weights/best.pt")

# time start for performance measuring
start = time.time()

# predict board corners
board_polygon = model.predict(img_rgb, imgsz=TARGET_SIZE, conf=CONFIDENCE_THRESHOLD)
board_points = predict_board_layout(img_resized, board_polygon)

# drow board corners
for pt in board_points:
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(img_resized, (x, y), 5, (0, 0, 255), -1)  # czerwone kółko, wypełnione

# wrap board perspective
M = cv2.getPerspectiveTransform(board_points, dst_pts)
warped_image = cv2.warpPerspective(img_resized, M, (TARGET_SIZE, TARGET_SIZE))

# get board grid
squares = get_grid(warped_image)

# drow board grid - squares
for sq in squares:
    pts = np.array(sq, dtype=np.int32)
    cv2.polylines(warped_image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)


# predict figures
results = model1(img_rgb, imgsz=TARGET_SIZE, conf=CONFIDENCE_THRESHOLD)


# checking whether figure is in a square
def point_in_quad(px, py, quad):
    """Sprawdza, czy punkt (px,py) jest wewnątrz czworokąta quad."""
    quad = np.array(quad, dtype=np.float32)
    # używamy funkcji OpenCV
    return cv2.pointPolygonTest(quad, (px, py), False) >= 0


# mapping figures to didicated position on grid
piece_positions = {}  # Key: square index 0..63, value: list of pieces in this square

for i, sq in enumerate(squares):
    piece_positions[i] = []


for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
    # change figure position to X: middle, Y: 1/3
    x1, y1, x2, y2 = box
    px = (x1 + x2) / 2
    py = y1 + (y2 - y1) * 0.8


    # drow pieces on origin image
    cv2.circle(img_resized, (int(px), int(py)), 5, (0, 0, 255), -1)
    label = f"{int(cls)}:{conf:.2f}"
    cv2.putText(img_resized, label, (int(px) + 5, int(py) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # transforming positions to warped image
    pt = np.array([[px, py]], dtype="float32")
    pt_warped = cv2.perspectiveTransform(pt[None, :, :], M)  # shape (1,1,2)
    px_w, py_w = pt_warped[0,0]

    # checking in which square is piece
    for i, sq in enumerate(squares):
        if point_in_quad(px_w, py_w, sq):
            piece_positions[i].append((int(px_w), int(py_w), int(cls), float(conf)))
            break

    # drowing figure's points on warped image
    cv2.circle(warped_image, (int(px_w), int(py_w)), 5, (0,255,0), -1)

# calc time
pred_time = time.time() - start

#print(piece_positions)

from utils.drow_2d_board import draw_2d_board_raw

# drawing raw board with figures
board_img = draw_2d_board_raw(piece_positions, squares, board_size=TARGET_SIZE)

# combine and show all 3 views
combined = np.hstack((board_img, warped_image, img_resized))
combined = cv2.resize(combined, (0, 0), fx=0.5, fy=0.5)

cv2.imshow("2D Raw + Warped + Origin", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()


print(f"YOLO prediction time: {pred_time*1000:.2f} ms")
print(f"Total per frame: {(pred_time)*1000:.2f} ms (~{1/(pred_time):.2f} FPS)")

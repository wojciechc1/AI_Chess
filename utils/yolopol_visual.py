import cv2
import numpy as np

# Wczytaj obraz
img = cv2.imread("../Dataset1/train/images/IMG_2769.jpg")
h, w = img.shape[:2]
# Skalujemy obraz do 640x640
new_w, new_h = 640, 640
img_resized = cv2.resize(img, (new_w, new_h))

# Przykładowe dane YOLOv8-seg
line = "0 0.08810758590698242 0.267578125 0.629774252573649 0.185546875 0.957899252573649 0.498046875 0.3328992525736491 0.703125"

# Zamień string na listę floatów, pomiń class_id
coords = list(map(float, line.split()[1:]))

# Przekształć znormalizowane współrzędne na piksele w nowym rozmiarze
pts = []
for i in range(0, len(coords), 2):
    x = int(coords[i] * new_w)
    y = int(coords[i+1] * new_h)
    pts.append([x, y])

pts = np.array(pts, np.int32)
pts = pts.reshape((-1, 1, 2))

# Narysuj polygon na przeskalowanym obrazie
cv2.polylines(img_resized, [pts], isClosed=True, color=(0, 0, 255), thickness=3)

cv2.imshow("Polygon 640x640", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
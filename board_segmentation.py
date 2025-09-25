import cv2
from ultralytics import YOLO
import numpy as np


def group_lines_by_angle(lines, angle_thresh=np.pi/8):
    """
    Grupy linii według podobnego kąta.
    lines: lista (vx, vy, x0, y0)
    angle_thresh: próg w radianach (~22.5°)
    Zwraca: listę dwóch grup po 2 linie (poziome i pionowe)
    """
    angles = [np.arctan2(vy, vx) for vx, vy, x0, y0 in lines]
    groups = []

    for i, line in enumerate(lines):
        added = False
        for g in groups:
            # porównanie kąta z pierwszą linią w grupie
            if abs(angles[i] - angles[g[0]]) < angle_thresh:
                g.append(i)
                added = True
                break
        if not added:
            groups.append([i])

    # dopilnowanie, żeby były dokładnie 2 grupy po 2 linie
    assert len(groups) == 2 and all(len(g) == 2 for g in groups), "Problem z grupowaniem linii"

    return groups

def line_intersection(line1, line2, img_shape=None, margin=0):
    """
    Znajduje punkt przecięcia dwóch linii wektorowych (vx,vy,x0,y0)
    Opcjonalnie sprawdza, czy punkt leży w obrazie z marginesem.
    """
    vx1, vy1, x01, y01 = line1
    vx2, vy2, x02, y02 = line2

    # równania parametryczne
    A = np.array([[vx1, -vx2], [vy1, -vy2]])
    b = np.array([x02 - x01, y02 - y01])

    if np.linalg.matrix_rank(A) < 2:
        return None  # linie równoległe

    t = np.linalg.solve(A, b)
    intersection = np.array([x01, y01]) + t[0] * np.array([vx1, vy1])

    if img_shape is not None:
        h, w = img_shape[:2]
        x, y = intersection
        if not (-margin <= x <= w + margin and -margin <= y <= h + margin):
            return None  # poza dozwolonym obszarem

    return intersection

def get_corners_from_groups(lines, groups):
    """
    lines: lista (vx, vy, x0, y0)
    groups: wynik z group_lines_by_angle()
    Zwraca 4 rogi jako listę punktów np. [[x1,y1], ...]
    """
    g1, g2 = groups
    corners = []

    for i in g1:
        for j in g2:
            inter = line_intersection(lines[i], lines[j])
            if inter is not None:
                corners.append(inter)

    return np.array(corners, dtype=np.float32)

def order_points(pts):
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:,1]-center[1], pts[:,0]-center[0])
    sorted_idx = np.argsort(angles)
    return pts[sorted_idx]

def best_fit_line_ransac(points, threshold=5, max_iters=1000):
    best_inliers = []
    best_line = None
    N = len(points)

    for _ in range(max_iters):
        idx = np.random.choice(N, 2, replace=False)
        p1, p2 = points[idx]
        if np.all(p1 == p2):
            continue
        [vx, vy, x0, y0] = [v[0] for v in cv2.fitLine(np.array([p1, p2]), cv2.DIST_L2, 0, 0.01, 0.01)]
        dists = np.abs((points - np.array([x0, y0])) @ np.array([-vy, vx]))
        inliers = points[dists < threshold]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_line = (vx, vy, x0, y0)
    return best_line, best_inliers


def predict_board_layout(img_resized, results):
    """
    Testuje model YOLOv8 segmentacji i rysuje wykryte maski/polygony.
    """


    for result in results:
        if hasattr(result, 'masks') and result.masks is not None:
            polygons = result.masks.xy
            for poly in polygons:
                poly_points = np.array(poly)  # punkty polygonu
                lines = []
                for _ in range(4):
                    line, inliers = best_fit_line_ransac(poly_points, threshold=5)
                    if line is None:
                        break
                    lines.append(line)
                    # usuwamy punkty, które już pasują do tej linii
                    poly_points = np.array([p for p in poly_points if not any(np.all(p == i) for i in inliers)])



                angles = []
                for (vx, vy, x0, y0) in lines:
                    x1, y1 = int(x0 - 1000 * vx), int(y0 - 1000 * vy)
                    x2, y2 = int(x0 + 1000 * vx), int(y0 + 1000 * vy)
                    cv2.line(img_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    angle = np.arctan2(vy, vx)
                    angles.append(angle)



                groups = group_lines_by_angle(lines)
                corners = get_corners_from_groups(lines, groups)
                print(corners)
                corners = order_points(corners)




                if len(corners) == 4:

                    src_pts = order_points(corners)

                    return src_pts







def get_grid(warped_image):
    N = 8  # 8x8 plansza
    h, w = warped_image.shape[:2]

    # Linie poziome i pionowe
    rows = np.linspace(0, h, N + 1, dtype=int)
    cols = np.linspace(0, w, N + 1, dtype=int)

    # Tworzenie pól 8x8
    squares = []
    for i in range(N):
        for j in range(N):
            top_left = (cols[j], rows[i])
            top_right = (cols[j + 1], rows[i])
            bottom_right = (cols[j + 1], rows[i + 1])
            bottom_left = (cols[j], rows[i + 1])
            squares.append([top_left, top_right, bottom_right, bottom_left])

    return squares

def draw_grid(warped_image, squares):
    for sq in squares:
        pts = np.array(sq, dtype=np.int32)
        cv2.polylines(warped_image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    # 2. Detransformacja siatki z powrotem na oryginał
    #M_inv = np.linalg.inv(M)
    #h_img, w_img = img_resized.shape[:2]
    #reprojected_grid = cv2.warpPerspective(img_copy, M_inv, (w_img, h_img))


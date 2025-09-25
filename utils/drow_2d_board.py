import cv2
import numpy as np


def draw_2d_board_raw(piece_positions, squares, board_size=640):
    """
    Rysuje techniczny widok 2D planszy z numerami klas pionków.

    piece_positions: dict, klucz = indeks pola 0..63, wartość = lista pionków (px, py, cls, conf)
    squares: lista 64 kwadratów (każdy = 4 punkty)
    board_size: rozmiar końcowego obrazu w px (kwadrat)
    """
    # Pusta plansza
    board_img = np.ones((board_size, board_size, 3), dtype=np.uint8) * 255

    # Rysowanie pól w szachownicę
    for i, sq in enumerate(squares):
        pts = np.array(sq, dtype=np.int32)
        color = (230, 230, 230) if (i // 8 + i % 8) % 2 == 0 else (180, 180, 180)
        cv2.fillPoly(board_img, [pts], color)
        cv2.polylines(board_img, [pts], isClosed=True, color=(0, 0, 0), thickness=1)

    # Rysowanie pionków jako numerów klas
    for idx, pieces in piece_positions.items():
        if pieces:
            cx = int(np.mean([p[0] for p in pieces]))
            cy = int(np.mean([p[1] for p in pieces]))
            class_idx = pieces[0][2]  # bierzemy najwyższe conf
            cv2.circle(board_img, (cx, cy), 15, (0, 255, 0), -1)
            cv2.putText(board_img, str(class_idx), (cx - 7, cy + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return board_img

# Przykład użycia:
# board_img = draw_2d_board_raw(piece_positions, squares)
# cv2.imshow("2D Board Raw", board_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

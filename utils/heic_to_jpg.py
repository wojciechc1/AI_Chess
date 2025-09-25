import os
import pillow_heif
from PIL import Image

# Ścieżka do folderu z plikami HEIC
input_folder = "./dataset/raw"
# Folder docelowy dla JPG (może być ten sam)
output_folder = "./dataset/processed-data"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".heic"):
        heic_path = os.path.join(input_folder, filename)
        jpg_filename = os.path.splitext(filename)[0] + ".jpg"
        jpg_path = os.path.join(output_folder, jpg_filename)

        # Rejestracja dekodera HEIF w Pillow
        pillow_heif.register_heif_opener()

        img = Image.open(heic_path)
        img.save(jpg_path, "JPEG")
        print(f"Skonwertowano: {filename} -> {jpg_filename}")

print("Konwersja zakończona!")



'''
import os

# Folder z labelami
folder_labels = "./y"

# Mapa starych numerów -> nowe (0–11 dla szachowych figur)
mapa_klas = {
    15: 0,  # white_pawn
    16: 6,  # black_pawn
    17: 1,  # white_knight
    18: 2,  # black_knight
    19: 3,  # white_bishop
    20: 4,  # black_bishop
    21: 5,  # white_rook
    22: 7,  # black_rook
    23: 8,  # white_queen
    24: 9,  # black_queen
    25: 10,  # white_king
    26: 11  # black_king
}

for filename in os.listdir(folder_labels):
    if filename.endswith(".txt"):
        path = os.path.join(folder_labels, filename)
        with open(path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            old_class = int(parts[0])
            if old_class in mapa_klas:
                new_class = mapa_klas[old_class]
                new_lines.append(" ".join([str(new_class)] + parts[1:]))
            else:
                # ignorujemy wszystkie inne klasy
                continue

        with open(path, "w") as f:
            f.write("\n".join(new_lines))
        print(f"Poprawiono: {filename}")

print("Wszystkie labelki szachowe są teraz w 0–11 i gotowe do YOLO!")
'''
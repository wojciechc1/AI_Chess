import json
import os

# Ścieżki
coco_json_path = "labels.json"  # Twój COCO JSON
images_dir = "images"
labels_dir = "labels"

os.makedirs(labels_dir, exist_ok=True)

# Wczytaj JSON
with open(coco_json_path, 'r') as f:
    data = json.load(f)

# Kategorie (tylko 1 klasa "board")
categories = {cat['id']: idx for idx, cat in enumerate(data.get('categories', [{'id':1,'name':'board'}]))}

# Przejdź przez wszystkie obrazy
for img in data['images']:
    img_id = img['id']
    img_name = img['file_name']
    img_w, img_h = img['width'], img['height']

    # Stwórz plik .txt w folderze labels
    label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
    lines = []

    # Znajdź wszystkie anotacje dla tego obrazu
    anns = [ann for ann in data['annotations'] if ann['image_id'] == img_id]

    for ann in anns:
        cat_id = categories.get(ann['category_id'], 0)
        segmentation = ann['segmentation'][0]  # przyjmujemy 1 polygon
        coords = []

        # Zamień na wartości znormalizowane do 0-1
        for i in range(0, len(segmentation), 2):
            x = segmentation[i] / img_w
            y = segmentation[i+1] / img_h
            coords.extend([x, y])

        # Wiersz do YOLOv8-seg
        line = f"{cat_id} " + " ".join([str(c) for c in coords])
        lines.append(line)

    # Zapisz plik
    with open(label_path, 'w') as f:
        f.write("\n".join(lines))

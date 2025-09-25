import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

labels_dir = Path('../dataset/train/labels')

heatmap = np.zeros((640,640))
for f in labels_dir.glob('*.txt'):
    with open(f) as lab:
        for line in lab:
            _, x, y, bw, bh = map(float, line.split())
            cx, cy = int(x*640), int(y*640)
            heatmap[cy,cx] += 1

plt.imshow(heatmap, cmap='hot')
plt.title('Box center heatmap')
plt.colorbar()
plt.show()

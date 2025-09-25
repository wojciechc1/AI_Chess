import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

labels_dir = Path('../dataset/train/labels')

counts = Counter()
for f in labels_dir.glob('*.txt'):
    with open(f) as lab:
        for line in lab:
            cls = int(line.split()[0])
            counts[cls] += 1

plt.bar(counts.keys(), counts.values())
plt.xlabel('Class ID')
plt.ylabel('Instances')
plt.title('Class Histogram')
plt.show()

import numpy as np
import cv2
import random
from heatmap import Heatmap

h = Heatmap(1280, 720)
# Add a crowd
points = []
# Dense area 1
for _ in range(30):
    points.append((random.randint(300, 500), random.randint(200, 400), random.randint(50, 100)))
# Dense area 2
for _ in range(50):
    points.append((random.randint(700, 900), random.randint(300, 500), random.randint(80, 150)))
# Scattered
for _ in range(20):
    points.append((random.randint(0, 1280), random.randint(0, 720), random.randint(30, 200)))

h.add_points(points)
h.add_points(points) # simulate consecutive frames

frame = np.zeros((720, 1280, 3), dtype=np.uint8)
out = h.get_heatmap(frame)
cv2.imwrite("test_crowd_heat.jpg", out)

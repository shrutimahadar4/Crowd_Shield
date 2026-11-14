import cv2
import numpy as np
import time
from app import process_frame
from tracker import SORT
from heatmap import Heatmap

print("Starting Privacy Mock Test...")

# Create dummy frame
dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
# Draw a face shape to see if it triggers
cv2.circle(dummy_frame, (320, 240), 50, (200, 200, 200), -1)

# Dummy config
heatmap = Heatmap(640, 480)
tracker = SORT()

print("Testing Standard Mode...")
f1, h1, _, _ = process_frame(dummy_frame.copy(), tracker, heatmap, 0.2, 0.4, False, False, run_ai=False)

print("Testing Face Blur Mode...")
f2, h2, _, _ = process_frame(dummy_frame.copy(), tracker, heatmap, 0.2, 0.4, True, False, run_ai=False)

print("Testing Silhouette Mode...")
f3, h3, _, _ = process_frame(dummy_frame.copy(), tracker, heatmap, 0.2, 0.4, False, True, run_ai=False)

print("All modes executed without exceptions!")

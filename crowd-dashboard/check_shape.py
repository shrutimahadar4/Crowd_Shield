import cv2
import numpy as np

# Load the generated heatmap
img = cv2.imread('test_out.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to find the heatmap footprint
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Find contours to inspect the shape
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    print(f"Bounding Box: x={x}, y={y}, w={w}, h={h}")
    # Compute the ratio of contour area to bounding box area to see if it's a rectangle
    area = cv2.contourArea(cnt)
    print(f"Contour Area: {area}")
    print(f"Bounding Box Area: {w * h}")
    print(f"Rectangularity: {area / (w * h):.2f}")
else:
    print("No heat footprint found.")

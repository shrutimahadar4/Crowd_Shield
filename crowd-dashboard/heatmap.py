import numpy as np
import cv2

class Heatmap:
    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self.map    = np.zeros((height, width), dtype=np.float32)
        # Slightly slower decay = richer historical density accumulation
        self.decay  = 0.92

    def add_points(self, points_with_size):
        """
        Accept a list of (x, y, person_width_px, person_height_px) tuples.
        Adaptive Gaussian sigmas scale with the person's width and height
        separately to cover the entire body shape accurately.
        """
        # Cool down existing heat
        self.map *= self.decay

        for (x, y, person_w, person_h) in points_with_size:
            # Fix: Allow y to equal self.height and x to equal self.width since
            # they are safely clamped in the actual array slicing logic below.
            if not (0 <= y <= self.height and 0 <= x <= self.width):
                continue

            # Scale sigma to encompass the physical width and height
            sigma_x = min(max(int(person_w * 0.25), 20), 500)
            sigma_y = min(max(int(person_h * 0.25), 35), 500)

            # Build a local Gaussian kernel and stamp it onto the map
            ksize_x = sigma_x * 6 + 1  # Must be odd
            ksize_y = sigma_y * 6 + 1
            if ksize_x % 2 == 0:
                ksize_x += 1
            if ksize_y % 2 == 0:
                ksize_y += 1

            # Clamp the kernel to a region within the map bounds
            y0 = max(0, y - ksize_y // 2)
            y1 = min(self.height, y + ksize_y // 2 + 1)
            x0 = max(0, x - ksize_x // 2)
            x1 = min(self.width, x + ksize_x // 2 + 1)

            # Build a small meshgrid patch and compute Gaussian weights
            ys = np.arange(y0, y1) - y
            xs = np.arange(x0, x1) - x
            gx = np.exp(-0.5 * (xs / sigma_x) ** 2)
            gy = np.exp(-0.5 * (ys / sigma_y) ** 2)
            patch = np.outer(gy, gx)

            # Accumulate — stronger heat value per detection
            self.map[y0:y1, x0:x1] += patch * 5.0

    def get_heatmap(self, frame):
        """
        Render the density map as a colour overlay on the frame.
        Uses JET colormap with smooth gradient.
        """
        # Strong smoothing pass to blend discrete Gaussian stamps into a cohesive density field
        blurred = cv2.GaussianBlur(self.map, (0, 0), sigmaX=35, sigmaY=35)
        
        # We want sparse patches to be visible (not squashed to 0) while dense areas max out at RED.
        # Instead of taking the square root (which ruins the Gaussian curve), we just clip the
        # absolute maximum value that we use for normalization. This ensures both sparse 
        # people and huge masses are coloured nicely with perfect Gaussian gradients!
        max_val = np.max(blurred)
        
        if max_val > 0.0001:
            # Clip the blurred map to an upper bound (say, 80) so dense crowds max out gracefully 
            # while a single person (which peaks around 60) still gets a solid bright colour.
            capped_blurred = np.clip(blurred, 0, 80.0)
            norm = cv2.normalize(capped_blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            norm = np.zeros_like(blurred, dtype=np.uint8)

        # JET: blue -> cyan -> green -> yellow -> red
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        # Adaptive alpha: transparent in low-density, opaque in hotspots
        alpha = norm.astype(np.float32) / 255.0
        
        # Threshold to remove near-zero density (keep background clean)
        # 1.2 exponent gives a nice soft fade at the outer Gaussian edges
        alpha = np.power(alpha, 1.2)
        
        # Increase the max alpha limit to provide richer colours for dense zones
        alpha = np.clip(alpha, 0, 0.65)

        alpha_3ch = np.dstack([alpha, alpha, alpha])
        blended = (frame * (1.0 - alpha_3ch) + colored * alpha_3ch).astype(np.uint8)

        return blended
"""
tracker.py  –  CrowdShield Pro | Kalman-Filter SORT Tracker
------------------------------------------------------------
A full SORT (Simple Online and Realtime Tracking) implementation
with Kalman Filters for smooth motion prediction, Hungarian-algorithm
assignment for robust matching, and sequential ID management.

Each person entering the frame receives a unique ID (1, 2, 3 …).
When that person leaves, the ID is "retired" — it is NEVER reused.
The  next  person that enters gets the next number in sequence.
This gives you:
    • Current frame count  = number of active tracks
    • Total unique persons = (next_id - 1)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


# ──────────────────────────────────────────────────────────────────
#  IoU helper
# ──────────────────────────────────────────────────────────────────
def _iou(box1, box2):
    """Return Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter   = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union  = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


# ──────────────────────────────────────────────────────────────────
#  Individual Track  (one person)
# ──────────────────────────────────────────────────────────────────
class KalmanTrack:
    """
    Wraps a 7-state Kalman filter for bounding-box tracking.

    State vector  x = [cx, cy, s, r, cx', cy', s']
        cx, cy  – centre of box
        s       – area (scale)
        r       – aspect ratio (width / height),  kept ~constant
        primed  – velocities of cx, cy, s

    Measurement  z = [cx, cy, s, r]
    """

    def __init__(self, bbox):
        self.id      = None       # assigned upon confirmation
        self.age     = 0          # total frames since creation
        self.hits    = 1          # consecutive matched frames
        self.no_match_streak = 0  # consecutive UNmatched frames

        # ── Kalman Filter (dim_x=7, dim_z=4) ──────────────────────
        kf = KalmanFilter(dim_x=7, dim_z=4)

        # Transition matrix  (constant-velocity model)
        kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=float)

        # Measurement matrix  (only cx, cy, s, r are observed)
        kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=float)

        # Measurement noise
        kf.R[2:, 2:] *= 10.0

        # Covariance – large uncertainty in initial velocities
        kf.P[4:, 4:] *= 1000.0
        kf.P         *= 10.0

        # Process noise
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01

        kf.x[:4] = self._to_z(bbox)
        self.kf = kf

    # ── helpers ───────────────────────────────────────────────────
    @staticmethod
    def _to_z(bbox):
        """[x1,y1,x2,y2]  →  [cx, cy, s, r]  (column vector)."""
        x1, y1, x2, y2 = bbox
        w   = x2 - x1
        h   = y2 - y1
        cx  = x1 + w / 2
        cy  = y1 + h / 2
        s   = w * h          # area
        r   = w / float(h) if h > 0 else 1.0
        return np.array([[cx], [cy], [s], [r]], dtype=float)

    @staticmethod
    def _to_bbox(x):
        """[cx, cy, s, r, …]  →  [x1, y1, x2, y2]."""
        x_flat = np.asarray(x).flatten()
        cx, cy, s, r = float(x_flat[0]), float(x_flat[1]), float(x_flat[2]), float(x_flat[3])
        if s < 0:
            s = 0.0
        w = np.sqrt(s * r) if r > 0 else 0.0
        h = s / w if w > 0 else 0.0
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [int(x1), int(y1), int(x2), int(y2)]

    # ── public API ────────────────────────────────────────────────
    def predict(self):
        """Advance the state estimate (call once per frame, before update)."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        if self.no_match_streak > 0:
            self.hits = 0

    def update(self, bbox):
        """Incorporate a new detection measurement."""
        self.no_match_streak = 0
        self.hits += 1
        self.kf.update(self._to_z(bbox))

    def get_bbox(self):
        """Return current best-estimate bounding box."""
        return self._to_bbox(self.kf.x)


# ──────────────────────────────────────────────────────────────────
#  SORT  –  the tracker
# ──────────────────────────────────────────────────────────────────
class SORT:
    """
    Simple Online and Realtime Tracker (Bewley et al. 2016).

    Parameters
    ----------
    max_age   : frames a track may persist without a matching detection
    min_hits  : detections needed before a track is considered confirmed
    iou_threshold : minimum IoU to associate a detection with a track
    """

    def __init__(self, max_age=60, min_hits=2, iou_threshold=0.20):
        self.max_age        = max_age
        self.min_hits       = min_hits
        self.iou_threshold  = iou_threshold
        self.tracks: list[KalmanTrack] = []
        self.confirmed_count = 0

    # ── frame update ──────────────────────────────────────────────
    def update(self, detections):
        """
        Parameters
        ----------
        detections : list of [x1, y1, x2, y2]

        Returns
        -------
        active : list of dicts  { 'id': int, 'bbox': [x1,y1,x2,y2] }
                 Only confirmed tracks (hits >= min_hits) are returned.
        """
        # 1. Predict new positions for every existing track
        for t in self.tracks:
            t.predict()

        # 2. Build cost matrix (1 – IoU) and run Hungarian assignment
        if self.tracks and detections:
            pred_boxes = [t.get_bbox() for t in self.tracks]
            cost = np.zeros((len(self.tracks), len(detections)), dtype=float)
            for ti, pb in enumerate(pred_boxes):
                for di, det in enumerate(detections):
                    cost[ti, di] = 1.0 - _iou(pb, det)

            # 3. Hungarian assignment
            row_idx, col_idx = linear_sum_assignment(cost)

            matched_tracks = set()
            matched_dets   = set()

            for r, c in zip(row_idx, col_idx):
                if cost[r, c] <= (1.0 - self.iou_threshold):
                    self.tracks[r].update(detections[c])
                    matched_tracks.add(r)
                    matched_dets.add(c)

            # 4. Increment no_match_streak for unmatched existing tracks
            for ti, track in enumerate(self.tracks):
                if ti not in matched_tracks:
                    track.no_match_streak += 1

            # 5. Create new tracks for unmatched detections
            for di, det in enumerate(detections):
                if di not in matched_dets:
                    self.tracks.append(KalmanTrack(det))

        elif detections:
            # ── No existing tracks but we have detections ──────────
            # Create fresh tracks. DO NOT age them here — they are
            # brand new (no_match_streak stays 0 so they show up
            # immediately on the very first frame / image upload).
            for det in detections:
                self.tracks.append(KalmanTrack(det))

        else:
            # ── Existing tracks but no detections this frame ───────
            for track in self.tracks:
                track.no_match_streak += 1

        # 6. Remove tracks that have been lost too long
        self.tracks = [t for t in self.tracks
                       if t.no_match_streak <= self.max_age]

        # 7. Return confirmed tracks only
        active = []
        for t in self.tracks:
            if t.hits >= self.min_hits:
                if t.id is None:
                    self.confirmed_count += 1
                    t.id = self.confirmed_count
                active.append({'id': t.id, 'bbox': t.get_bbox()})

        return active

    @property
    def total_unique(self):
        """Total distinct persons confirmed since tracker was created."""
        return self.confirmed_count

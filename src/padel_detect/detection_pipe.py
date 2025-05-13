import os
import cv2
import numpy as np
from padel_detect import Detector


def process_input(path: str,
                  detector: Detector,
                  proximity_thresh: float = 0.1):
    """
    Process either an image or a video file for padel detection using normalized coordinates.
    Returns a dict mapping frame_id (or 0 for image) to detection dict or empty dict.
    Detection dict contains 'player_id', 'keypoints', and 'ball'.
    :param proximity_thresh: maximum normalized distance (0-1) to accept a wrist-ball match
    """
    # Determine if input is image by extension
    _, ext = os.path.splitext(path)
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def select_detection(keypoints_list, ball_pos):
        # Both keypoints_list and ball_pos are in normalized [0,1] coords
        best = None
        best_dist = float('inf')
        for idx, kpts in enumerate(keypoints_list):
            if hasattr(kpts, '__len__') and len(kpts) >= 11:
                for side in (kpts[9], kpts[10]):
                    d = np.linalg.norm(np.array(side) - np.array(ball_pos))
                    if d < best_dist:
                        best_dist = d
                        best = (idx, kpts)
        if best is not None and best_dist <= proximity_thresh:
            idx, kpts = best
            return {
                'player_id': idx,
                'keypoints': kpts.tolist(),
                'ball': ball_pos,
            }
        return {}

    # Image case
    if ext.lower() in image_exts:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot read image file {path}")
        keypoints_list = detector.detect_keypoints(img)
        ball_pos = detector.detect_ball(img)
        detection = select_detection(keypoints_list, ball_pos) if ball_pos and keypoints_list else {}
        return {0: detection}

    # Video case
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {path}")
    frame_id = 0
    results = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keypoints_list = detector.detect_keypoints(frame)
        ball_pos = detector.detect_ball(frame)
        detection = select_detection(keypoints_list, ball_pos) if ball_pos and keypoints_list else {}
        results[frame_id] = detection
        frame_id += 1
    cap.release()
    return results

import cv2
import numpy as np
from padel_detection.detector import Detector
import os


def process_input(path: str,
                  detector: Detector,
                  proximity_thresh: float = 10.0):
    """
    Process either an image or video file.
    - If `path` is an image, runs detection on the single image and returns {0: detection}.
    - If `path` is a video, processes frame by frame as before.
    Returns dict mapping frame_id to detection dict or empty dict.
    """
    # Determine if path is image by extension
    ext = os.path.splitext(path)[1].lower()
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    # Single image
    if ext in image_exts:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot read image file {path}")
        keypoints_list = detector.detect_keypoints(img)
        ball_pos = detector.detect_ball(img)
        print(keypoints_list)
        print(ball_pos)
        detection = {}
        if ball_pos and keypoints_list:
            for idx, kpts in enumerate(keypoints_list):
                left_wrist = kpts[10]
                right_wrist = kpts[11]
                d_l = np.linalg.norm(left_wrist - ball_pos)
                d_r = np.linalg.norm(right_wrist - ball_pos)
                print('distance to left wrist: ', d_l)
                print('distance to right wrist: ', d_r)
                if d_l < proximity_thresh or d_r < proximity_thresh:
                    detection = {
                        'player_id': idx,
                        'keypoints': kpts.tolist(),
                        'ball': {'position': ball_pos}
                    }
                    break
        return {0: detection}

    # Otherwise, treat as video
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
        detection = {}
        if ball_pos and keypoints_list:
            for idx, kpts in enumerate(keypoints_list):
                left_wrist = kpts[10]
                right_wrist = kpts[11]
                d_l = np.linalg.norm(left_wrist - ball_pos)
                d_r = np.linalg.norm(right_wrist - ball_pos)
                if d_l < proximity_thresh or d_r < proximity_thresh:
                    detection = {
                        'player_id': idx,
                        'keypoints': kpts.tolist(),
                        'ball': {'position': ball_pos}
                    }
                    break
        results[frame_id] = detection
        frame_id += 1

    cap.release()
    return results
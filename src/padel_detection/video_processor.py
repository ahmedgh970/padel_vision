import cv2
import numpy as np
from padel_detection.detector import Detector

def process_video(path: str,
                  detector: Detector,
                  proximity_thresh: float = 10.0):
    """
    Iterate video frames, detect keypoints and ball proximity.
    Returns a dict mapping frame_id to a detection dict or empty dict.
    """
    cap = cv2.VideoCapture(path)
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
            print(f'==========> Frame {frame_id}: Sports ball detected and keypoints not empty')
            for idx, kpts in enumerate(keypoints_list):
                left_wrist = kpts[10]; right_wrist = kpts[11]
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
        print(f"Frame {frame_id} have been processed")

    cap.release()
    return results
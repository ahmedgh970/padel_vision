import os
import cv2
import numpy as np
from typing import List, Tuple


def draw_annotations(frame: np.ndarray, det: dict):
    # TODO: select only the unormalized coords
    # Draw ball
    bx, by = det['ball']
    cv2.circle(frame, (int(bx), int(by)), 6, (0, 0, 255), -1)
    # Draw keypoints
    keypoints = det['keypoints']
    for x, y in keypoints:
        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
    # Draw lines between keypoints if connections provided
    # TODO: Add the desired connections
    connections = []
    if connections:
        for i, j in connections:
            x1, y1 = keypoints[i]
            x2, y2 = keypoints[j]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

def annotate_and_save_frames(input_path: str,
                            detections: dict,
                            output_dir: str,):
    """
    For each frame of a video or a single image, if a valid detection is present,
    overlay ball position and keypoints (and optional connections) and save the frame.

    :param input_path: Path to video or image.
    :param detections: Dict mapping frame_id (or 0 for image) to detection dict.
    :param output_dir: Directory where annotated frames will be saved.
    :param connections: List of keypoint index pairs to connect with lines.
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(input_path)[1].lower()
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # Single image case
    if ext in image_exts:
        frame_id = 0
        frame = cv2.imread(input_path)
        if frame is None:
            raise ValueError(f"Cannot read image file {input_path}")
        det = detections.get(frame_id, {})
        if det:
            draw_annotations(frame, det)
            out_path = os.path.join(output_dir, f"frame_{frame_id:06d}.jpg")
            cv2.imwrite(out_path, frame)
        return

    # Video case
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {input_path}")
    for frame_id in sorted(detections.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Cannot read frame {frame_id} from video file {input_path}")
        det = detections[frame_id]
        draw_annotations(frame, det)
        out_path = os.path.join(output_dir, f"frame_{frame_id:06d}.jpg")
        cv2.imwrite(out_path, frame)
    cap.release()

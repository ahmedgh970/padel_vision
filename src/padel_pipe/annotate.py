import os
import cv2


def annotate_video(input_path: str, detections: dict, output_path: str):
    """
    Overlays detections on each frame and writes an annotated video.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        det = detections.get(frame_id, {})
        if det:
            # draw ball
            bx, by = det['ball']
            cv2.circle(frame, (int(bx), int(by)), 6, (0, 0, 255), -1)
            # draw wrists
            for idx, (x, y) in enumerate(det['keypoints']):
                cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)
        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
import cv2
import os

def sample_frames_to_video(input_path: str,
                           output_path: str,
                           num_frames: int,
                           start_sec: float = 0.0,
                           end_sec: float = None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) if end_sec else total_frames
    end_frame = min(end_frame, total_frames)
    if start_frame < 0 or start_frame >= end_frame:
        raise ValueError("Invalid start_sec/end_sec")
    segment_frames = end_frame - start_frame

    if num_frames >= segment_frames or num_frames <= 0:
        indices = list(range(start_frame, end_frame))
    else:
        step = segment_frames / float(num_frames)
        indices = [start_frame + int(step * i) for i in range(num_frames)]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    sampled_idx = 0
    target_idx = indices[sampled_idx] if indices else None

    while current_frame < end_frame and sampled_idx < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame == target_idx:
            out.write(frame)
            sampled_idx += 1
            if sampled_idx < len(indices):
                target_idx = indices[sampled_idx]
        current_frame += 1

    cap.release()
    out.release()
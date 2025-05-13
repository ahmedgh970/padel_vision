import cv2
import numpy as np
import pytest
from padel_pipe import annotate_video


def create_dummy_video(path: str, num_frames: int = 2, width: int = 64, height: int = 48, fps: int = 10):
    """
    Create a dummy video file with solid-color frames.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        # generate a solid color frame (changes over frames)
        frame = np.full((height, width, 3), fill_value=(i * 50) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()


def test_annotate_video_creates_file(tmp_path):
    # Arrange
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "annotated.mp4"
    # Create a 3-frame dummy video
    create_dummy_video(str(input_path), num_frames=3)
    # Detections: annotate a ball at center of frame 1 only
    detections = {
        0: {},
        1: {
            'ball': (32, 24),
            'keypoints': [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50),
                          (15, 15), (25, 25), (35, 35), (45, 45), (55, 55),
                          (60, 60), (65, 65), (70, 70), (75, 75), (80, 80),
                          (85, 85), (90, 90)]
        },
        2: {}
    }

    # Act
    annotate_video(str(input_path), detections, str(output_path))

    # Assert file exists and frame count matches
    assert output_path.exists(), "Annotated video file was not created"
    cap = cv2.VideoCapture(str(output_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert count == 3, f"Expected 3 frames, got {count}"


def test_annotate_invalid_input(tmp_path):
    # Invalid input path should raise ValueError
    with pytest.raises(ValueError):
        annotate_video(str(tmp_path / "nonexistent.mp4"), {}, str(tmp_path / "out.mp4"))
